import torch
from torch import nn
class MGL(nn.Module):
    def __init__(self, in_features, num_moments, hidden_features=None, out_features=None):
        super().__init__()

        # params
        self.in_features = in_features
        assert num_moments >= 2
        self.num_moments = num_moments
        self.out_features = (in_features * num_moments if not out_features else out_features)
        self.hidden_features = (in_features if not hidden_features else hidden_features)

        # layers
        self.linear_in = nn.Linear(self.in_features, self.hidden_features)
        self.linear_out = nn.Linear(self.num_moments*self.hidden_features, self.out_features)

    @staticmethod
    def _centered_moment(x_centered, x_std, t):
        return torch.div((x_centered**t).mean(1, keepdim=True), x_std**t) #.squeeze()
        # return torch.div((x_centered**t).mean(1, keepdim=True), x_std**t).squeeze()

    def forward(self, input):
        """
        I used standardized moment for stability. (https://en.wikipedia.org/wiki/Standardized_moment)
        Can be further improved using masking i.e. it should work with varying sequence lenght.
        (See if we can compute on standardized data instead of centered data.)
        """
        x, batch_sizes, sorted_indices, unsorted_indices = input
        import pdb; pdb.set_trace()
        # x = x.unsqueeze(0)
        import pdb; pdb.set_trace()
        x = self.linear_in(x) # [B, N, D_in] -> [B, N, H]

        x_mean = x.mean(1, keepdim=True)

        x_std = x.std(1, unbiased=True, keepdim=True)

        x_centered = x - x_mean
        # https://stackoverflow.com/questions/65993928/indexerror-dimension-out-of-range-pytorch-dimension-expected-to-be-in-range-o
        calced_moments = [x_mean, x_std] + [self._centered_moment(x_centered, x_std, t) for t in range(3, self.num_moments+1)]

        # calced_moments = [x_mean.squeeze(), x_std.squeeze()] + [
        #     self._centered_moment(x_centered, x_std, t) for t in range(3, self.num_moments+1)]
        x = torch.cat(calced_moments, 1) # [B, N, H] -> [B, M*H]
        # x = torch.cat(calced_moments, 0) # [B, N, H] -> [B, M*H]
        # x = torch.cat([x_mean.squeeze(), x_std.squeeze()] + [self._centered_moment(x_centered, x_std, t) for t in range(3, self.num_moments+1)], 1) # [B, N, H] -> [B, M*H]

        import pdb; pdb.set_trace()
        x = self.linear_out(x) # [B, M*H] -> [B, D_out]

        return x