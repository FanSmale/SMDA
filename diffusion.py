import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pylab as plt
from tqdm import tqdm
import gif


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

    :param v:           Coefficients vector (Covering all time steps)
    :param t:           Specific time steps
    :param x_shape:     Image size (including batch, with four dimensions)
    :return:            Coefficients corresponding to specific time steps (Tensor)
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def space_timesteps(num_timesteps, section_counts):
    """
    For example, if there are 300 timesteps and the section counts are [10,15,20],
    then the first 100 timesteps are strided to be 10 timesteps,
    the second 100 are strided to be 15 timesteps, and the final 100 are strided to be 20.

    :param num_timesteps:   The original time step range
    :param section_counts:  Split prompt keywords
                            (None, list or str, if it is a str, it is represented by "ddpmXX" or "ddimXX", where
                            XX represents the reduced step size)
    :return:                Reduced time step list
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim") or section_counts.startswith("ddpm"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class GaussianDiffusionTrainer(nn.Module):
    """
    Part of the diffusion model used for training.
    """
    def __init__(self, model, beta_1, beta_T, T):
        """
        Construtor

        :param model:           Network model (unet)
        :param beta_1:          \beta_1
        :param beta_T:          \beta_T
        :param T:               Time step range
        """
        super().__init__()

        self.model = model
        self.T = T

        # Prepare \beta, \alpha and \bar{\alpha} for all time steps
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    # def get_xt_by_t_x0(self, x_0: torch.Tensor, t: torch.Tensor):
    #     noise = torch.randn_like(x_0)
    #     return  extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
    #             extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise

    def forward(self, x_0, n_class, zsem):
        """
        Training eps network (Algorithm 1 in DDPM paper)

        :param x_0:             Clean image
        :param n_class:         Category information vector
        :param zsem:            Semantic latent code
        :return:                Loss (l2) between random noise and predicted noise
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 \
            + extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.mse_loss(self.model(x_t, t, n_class, zsem), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    """
    Part of the diffusion model used for generation and reverse generation. (DDIM)
    """
    def __init__(self, model, beta_1, beta_T, T, section_counts=None):
        """
        Construtor

        :param model:           Network model (unet)
        :param beta_1:          \beta_1
        :param beta_T:          \beta_T
        :param T:               Time step range
        :param section_counts:  Split prompt keywords
                                (here it must be str or None, it is represented by "ddimXX", where XX represents the
                                 reduced step size)
        """
        super().__init__()

        self.model = model
        self.T = T
        self.section_counts = section_counts

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, self.T).double())

        self.num_timesteps = None
        temp_list = []
        if self.section_counts is not None:
            self.num_timesteps = space_timesteps(num_timesteps=self.T, section_counts=self.section_counts)
            for i in range(self.T):
                if i in self.num_timesteps:
                    temp_list.append(self.betas[i])
            self.betas = torch.tensor(temp_list).to(self.betas.device)

        # Prepare \beta, \alpha and \bar{\alpha} for all time steps
        alphas = 1. - self.betas
        self.register_buffer('alphas_bar', torch.cumprod(alphas, dim=0))
        self.register_buffer('alphas_bar_prev', F.pad(self.alphas_bar, [1, 0], value=1.0)[:len(self.betas)])
        self.register_buffer('alphas_bar_next', F.pad(self.alphas_bar, [0, 1], value=0.0)[1:])

        # DDIM related parameters
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1. / self.alphas_bar))
        self.register_buffer('sqrt_recipm1_alphas_bar', torch.sqrt(1. / self.alphas_bar - 1))

    # def predict_xt_prev_mean_from_eps_with_ddpm(self, x_t, t, eps):
    #     assert x_t.shape == eps.shape
    #     return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps
    #
    # def predict_xt_prev_mean_from_eps_with_ddim(self, x_t, t, eps):
    #     assert x_t.shape == eps.shape
    #     return extract(self.coeff3, t, x_t.shape) * x_t - extract(self.coeff4, t, x_t.shape) * eps + \
    #            extract(self.coeff5, t, x_t.shape) * eps

    # def p_mean_variance_with_ddpm(self, x_t, t, n_class, zsem):
    #     # below: only log_variance is used in the KL computations
    #     var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
    #     var = extract(var, t, x_t.shape)
    #     eps = self.model(x_t, t, n_class, zsem)
    #     non_eps = self.model(x_t, t, torch.zeros_like(n_class).to(n_class.device), zsem)
    #
    #     eps = (1. + self.w) * eps - self.w * non_eps
    #     xt_prev_mean = self.predict_xt_prev_mean_from_eps_with_ddpm(x_t, t, eps=eps)
    #     return xt_prev_mean, var

    # def p_mean_variance_with_ddim(self, x_t, t, n_class, zsem):
    #     # get var
    #     var = self.sigma_squared
    #     var = extract(var, t, x_t.shape)
    #
    #     # get network prediction
    #     eps = self.model(x_t, t, n_class, zsem)
    #     non_eps = self.model(x_t, t, torch.zeros_like(n_class).to(n_class.device), zsem)
    #     eps = (1. + self.w) * eps - self.w * non_eps
    #
    #     # get mean
    #     xt_prev_mean = self.predict_xt_prev_mean_from_eps_with_ddim(x_t, t, eps=eps)
    #
    #     return xt_prev_mean, var

    def predict_xstart_from_eps(self, x_t, t, eps):
        """
        f^\textbf{D}_\theta

        :param x_t:             Image with t level noise
        :param t:               Time step
        :param eps:             Prior noise
        :return:                The prior prediction $\hat{x}_{0|t}$ of the trained network $\epsilon_\theta(x_t; t)$
                                for $x_0$.
        """
        assert x_t.shape == eps.shape
        return (extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps)

    # def predict_eps_from_xstart(self, x_t, t, xstart):
    #     return (extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t - xstart) \
    #             / extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape)

    def ddim_reverse_sample(self, x_t, t, n_class, zsem):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.

        :param x_t:             Image with t level noise
        :param t:               Time step
        :param n_class:         Category information
        :param zsem:            Semantic latent code
        :return:                x_{t+1}
        """

        eps_predicted_by_net = self.model(x_t, t, n_class, zsem)
        xstart = self.predict_xstart_from_eps(x_t, t, eps=eps_predicted_by_net)

        eps = (extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t - xstart) /\
               extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape)

        alpha_bar_next = extract(self.alphas_bar_next, t, x_t.shape)

        mean_pred = (xstart * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps)

        return mean_pred

    def ddim_reverse_sample_loop(self, forward_T, x_0, n_class, zsem, save_gif_path=None, use_tqdm=False):
        """
        Execute DDIM reverse iteration

        :param forward_T:       Iteration step
        :param x_0:             Clean image
        :param n_class:         Category information
        :param zsem:            Semantic latent code
        :param save_gif_path:   The save path of the dynamic process GIF file (opt.)
        :param use_tqdm:        Whether to display tqdm
        :return:                x^D_T
        """

        x_p = x_0.clone().detach()
        frames = []

        if use_tqdm:
            with tqdm(range(forward_T)) as tr_obj:
                for i in tr_obj:
                    tr_obj.set_postfix(ordered_dict={"step": i})

                    t = x_p.new_ones([x_0.shape[0], ], dtype=torch.long, device=x_0.device) * i
                    x_p = self.ddim_reverse_sample(x_t=x_p, t=t, n_class=n_class, zsem=zsem)
                    if save_gif_path is not None:
                        @gif.frame
                        def pain_img(index):
                            plt.imshow(x_p[0][0].cpu())
                            plt.title("index: {}".format(index))
                        frames.append(pain_img(i))
        else:
            for i in range(forward_T):
                t = x_p.new_ones([x_0.shape[0], ], dtype=torch.long, device=x_0.device) * i
                x_p = self.ddim_reverse_sample(x_t=x_p, t=t, n_class=n_class, zsem=zsem)
                if save_gif_path is not None:
                    @gif.frame
                    def pain_img(index):
                        plt.imshow(x_p[0][0].cpu())
                        plt.title("index: {}".format(index))

                    frames.append(pain_img(i))
        if save_gif_path is not None:
            gif.save(frames, save_gif_path, duration=0.0)

        return x_p

    def ddim_sample(self, x_t, t, n_class, zsem, eta):
        """
        Sample x_{t-1} from the model using DDIM.

        :param x_t:             Image with t level noise
        :param t:               Time step
        :param n_class:         Category information
        :param zsem:            Semantic latent code
        :param eta:             Hyperparameters controlling the sampling mechanism (eta=0 for DDIM)
        :return:                x_{t-1}
        """

        eps_predicted_by_net = self.model(x_t, t, n_class, zsem)
        xstart = self.predict_xstart_from_eps(x_t, t, eps=eps_predicted_by_net)

        alpha_bar = extract(self.alphas_bar, t, x_t.shape)
        alpha_bar_prev = extract(self.alphas_bar_prev, t, x_t.shape)
        sigma = (eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev))

        noise = torch.randn_like(x_t)
        mean_pred = (xstart * torch.sqrt(alpha_bar_prev) +
                     torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps_predicted_by_net)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))               # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return sample

    def ddim_sample_loop(self, backend_T, x_t, n_class, zsem, save_gif_path=None, use_tqdm=False):
        """
        Execute DDIM forward iteration

        :param backend_T:       Iteration step
        :param x_t:             Image with t level noise
        :param n_class:         Category information
        :param zsem:            Semantic latent code
        :param save_gif_path:   The save path of the dynamic process GIF file (opt.)
        :param use_tqdm:        Whether to display tqdm
        :return:                x^D_0
        """
        x_p = x_t.clone().detach()

        re_list = list(reversed(range(backend_T)))
        frames = []

        if use_tqdm:
            with tqdm(range(backend_T)) as tr_obj:
                for k in tr_obj:
                    i = re_list[k]
                    tr_obj.set_postfix(ordered_dict={"step": i})

                    t = x_p.new_ones([x_t.shape[0], ], dtype=torch.long, device=x_t.device) * i
                    x_p = self.ddim_sample(x_t=x_p, t=t, n_class=n_class, zsem=zsem, eta=0)
                    if save_gif_path is not None:
                        @gif.frame
                        def pain_img(index):
                            plt.imshow(x_p[0][0].cpu())
                            plt.title("index: {}".format(index))

                        frames.append(pain_img(i))
        else:
            for k in range(backend_T):
                i = re_list[k]

                t = x_p.new_ones([x_t.shape[0], ], dtype=torch.long, device=x_t.device) * i
                x_p = self.ddim_sample(x_t=x_p, t=t, n_class=n_class, zsem=zsem, eta=0)
                if save_gif_path is not None:
                    @gif.frame
                    def pain_img(index):
                        plt.imshow(x_p[0][0].cpu())
                        plt.title("index: {}".format(index))

                    frames.append(pain_img(i))

        if save_gif_path is not None:
            gif.save(frames, save_gif_path, duration=0.0)

        return x_p

    # def forward(self, x_T, n_class, zsem):
    #     """
    #     Algorithm 2.
    #     """
    #     x_t = x_T
    #
    #     for time_step in reversed(range(len(self.betas))):
    #         print(time_step)
    #         t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
    #
    #         if self.section_counts is None or self.section_counts.startswith("ddpm"):
    #             mean, var = self.p_mean_variance_with_ddpm(x_t=x_t, t=t, n_class=n_class, zsem=zsem)
    #         elif self.section_counts.startswith("ddim"):
    #             mean, var = self.p_mean_variance_with_ddim(x_t=x_t, t=t, n_class=n_class, zsem=zsem)
    #         else:
    #             print("Parameter section_counts is wrong!")
    #             exit(0)
    #
    #         if time_step > 0:
    #             noise = torch.randn_like(x_t)
    #         else:
    #             noise = 0
    #
    #         x_t = mean + torch.sqrt(var) * noise
    #         assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
    #     x_0 = x_t
    #     return torch.clip(x_0, -1, 1)




