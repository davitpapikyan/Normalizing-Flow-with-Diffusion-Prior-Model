import unittest

import torch

from normalizing_flow import InvConv2d, InvConv2dLU, ActNorm, AffineCoupling


class TestActNorm(unittest.TestCase):
    """Testing activation normalization layer."""

    @torch.no_grad()
    def setUp(self):
        """Setting up testing objects."""
        self.x = torch.randn(size=(32, 3, 28, 28))
        self.f = ActNorm(in_channels=self.x.size(1))

        self.y, self.logdet = self.f.transform(self.x)
        self.inv_y, self.logdet_inv = self.f.invert(self.y)

        self.eps = 1e-4

    @torch.no_grad()
    def test_inverse(self):
        """Testing the inverse operation."""
        self.assertTrue(self.x.size() == self.y.size(),
                        msg="x and y are supposed to have the same dimensions.")

        self.assertEqual((self.inv_y - self.x).norm() < self.eps, torch.tensor(True),
                         msg="inv_y and x are supposed to be close to each other.")

    @torch.no_grad()
    def test_output(self):
        """Testing the statistics of the forward transformation."""
        self.assertEqual(self.y.mean(dim=(0, 2, 3)).norm() < self.eps, torch.tensor(True),
                         msg="y is supposed to have zero mean per channel.")

        self.assertEqual((self.y.mean(dim=(0, 2, 3)) - torch.tensor([0.0, 0.0, 0.0])).abs().max() < self.eps,
                         torch.tensor(True), msg="y is supposed to have zero mean per channel.")

        self.assertEqual((self.y.var(dim=(0, 2, 3)) - torch.tensor([1.0, 1.0, 1.0])).norm() < self.eps,
                         torch.tensor(True), msg="y is supposed to have unit variance per channel.")


class TestInvConv2d(unittest.TestCase):
    """Testing invertible 1x1 convolution."""

    @torch.no_grad()
    def setUp(self):
        """Setting up testing objects."""
        self.x = torch.randn(size=(32, 3, 28, 28))
        self.f = InvConv2d(in_channels=self.x.size(1))

        self.y, self.logdet = self.f.transform(self.x)
        self.inv_y, self.logdet_inv = self.f.invert(self.y)

        self.eps = 1e-4

    @torch.no_grad()
    def test_inverse(self):
        """Testing the inverse operation."""
        self.assertTrue(self.x.size() == self.y.size(),
                        msg="x and y are supposed to have the same dimensions.")

        self.assertEqual((self.inv_y - self.x).norm() < self.eps, torch.tensor(True),
                         msg="inv_y and x are supposed to be close to each other.")


class TestInvConv2dLU(unittest.TestCase):
    """Testing invertible 1x1 convolution with LU decomposition."""

    @torch.no_grad()
    def setUp(self):
        """Setting up testing objects."""
        self.x = torch.randn(size=(32, 3, 28, 28))
        self.f = InvConv2dLU(in_channels=self.x.size(1))

        self.y, self.logdet = self.f.transform(self.x)
        self.inv_y, self.logdet_inv = self.f.invert(self.y)

        self.eps = 1e-3

    @torch.no_grad()
    def test_inverse(self):
        """Testing the inverse operation."""
        self.assertTrue(self.x.size() == self.y.size(),
                        msg="x and y are supposed to have the same dimensions.")

        self.assertEqual((self.inv_y - self.x).norm() < self.eps, torch.tensor(True),
                         msg="inv_y and x are supposed to be close to each other.")

    @torch.no_grad()
    def test_jacobian(self):
        """Testing the calculation of log abs det jacobian."""
        weight = self.f.w_P @ self.f.w_L @ (self.f.w_U + torch.diag(self.f.w_s))
        logdet2 = self.x.size(2) * self.x.size(3) * torch.det(weight).abs().log()
        self.assertEqual((self.logdet - logdet2).norm() < self.eps, torch.tensor(True),
                         msg="logdet and logdet2 are supposed to be close to each other.")
        self.assertEqual((self.logdet_inv + logdet2).norm() < self.eps, torch.tensor(True),
                         msg="logdet_inv and -logdet2 are supposed to be close to each other.")


class TestAffCoupling(unittest.TestCase):
    """Testing affine coupling layer."""

    @torch.no_grad()
    def setUp(self):
        """Setting up testing objects."""
        self.x = torch.randn(size=(32, 4, 28, 28))
        self.f = AffineCoupling(self.x.size(1) // 2)

        self.y, self.logdet = self.f.transform(self.x)
        self.inv_y, self.logdet_inv = self.f.invert(self.y)

        self.eps = 1e-4

    @torch.no_grad()
    def test_forward(self):
        """Testing the forward transformation."""
        self.assertTrue(self.x.size() == self.y.size() == self.inv_y.size(),
                        msg="x, y and inv_y are supposed to have the same dimensions.")
        self.assertEqual((self.inv_y - self.x).norm() < self.eps, torch.tensor(True),
                         msg="inv_y and x are supposed to be close to each other.")


if __name__ == '__main__':
    unittest.main()
