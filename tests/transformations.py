import unittest

import torch

from normalizing_flow import InvConv2d, ActNorm, AffineCoupling

EPS = 1e-3


class TestActNorm(unittest.TestCase):
    """Testing activation normalization layer."""

    @torch.no_grad()
    def setUp(self):
        """Setting up testing objects."""
        self.x = torch.randn(size=(32, 3, 28, 28))
        self.f = ActNorm(in_channels=self.x.size(1))
        self.logdet, self.logp = torch.zeros(self.x.size(0)), torch.zeros(self.x.size(0))

        self.y, self.logdet, self.logp = self.f.transform(self.x, self.logdet, self.logp)
        self.inv_y = self.f.invert(self.y)

    @torch.no_grad()
    def test_inverse(self):
        """Testing the inverse operation."""
        self.assertTrue(self.x.size() == self.y.size(),
                        msg="x and y are supposed to have the same dimensions.")

        self.assertEqual((self.inv_y - self.x).norm() < EPS, torch.tensor(True),
                         msg="inv_y and x are supposed to be close to each other.")

    @torch.no_grad()
    def test_output(self):
        """Testing the statistics of the forward transformation."""
        self.assertEqual(self.y.mean(dim=(0, 2, 3)).norm() < EPS, torch.tensor(True),
                         msg="y is supposed to have zero mean per channel.")

        self.assertEqual((self.y.mean(dim=(0, 2, 3)) - torch.tensor([0.0, 0.0, 0.0])).abs().max() < EPS,
                         torch.tensor(True), msg="y is supposed to have zero mean per channel.")

        self.assertEqual((self.y.var(dim=(0, 2, 3)) - torch.tensor([1.0, 1.0, 1.0])).norm() < EPS,
                         torch.tensor(True), msg="y is supposed to have unit variance per channel.")


class TestInvConv2d(unittest.TestCase):
    """Testing invertible 1x1 convolution."""

    @torch.no_grad()
    def setUp(self):
        """Setting up testing objects."""
        self.x = torch.randn(size=(32, 3, 28, 28))
        self.f = InvConv2d(in_channels=self.x.size(1))
        self.logdet, self.logp = torch.zeros(self.x.size(0)), torch.zeros(self.x.size(0))

        self.y, self.logdet, self.logp = self.f.transform(self.x, self.logdet, self.logp)
        self.inv_y = self.f.invert(self.y)

    @torch.no_grad()
    def test_inverse(self):
        """Testing the inverse operation."""
        self.assertTrue(self.x.size() == self.y.size(),
                        msg="x and y are supposed to have the same dimensions.")
        self.assertEqual((self.inv_y - self.x).norm() < EPS, torch.tensor(True),
                         msg="inv_y and x are supposed to be close to each other.")


class TestAffCoupling(unittest.TestCase):
    """Testing affine coupling layer."""

    @torch.no_grad()
    def setUp(self):
        """Setting up testing objects."""
        self.x = torch.randn(size=(32, 4, 28, 28))
        self.f = AffineCoupling(self.x.size(1))
        self.logdet, self.logp = torch.zeros(self.x.size(0)), torch.zeros(self.x.size(0))

        self.y, self.logdet, self.logp = self.f.transform(self.x, self.logdet, self.logp)
        self.inv_y = self.f.invert(self.y)

    @torch.no_grad()
    def test_forward(self):
        """Testing the forward transformation."""
        self.assertTrue(self.x.size() == self.y.size() == self.inv_y.size(),
                        msg="x, y and inv_y are supposed to have the same dimensions.")
        self.assertEqual((self.inv_y - self.x).norm() < EPS, torch.tensor(True),
                         msg="inv_y and x are supposed to be close to each other.")


if __name__ == '__main__':
    unittest.main()
