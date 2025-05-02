import unittest
import torch

from loftq import BlockQuantizer
# Make sure to include the safe_broadcast_subtract definition here or import it

class TestSafeBroadcastSubtract(unittest.TestCase):

    def compare_with_standard(self, A, B):
        """Helper to compare safe_broadcast_subtract result with standard subtraction."""
        A_gpu = A.to('cuda')
        B_gpu = B.to('cuda')
        expected = torch.abs((A - B)).cpu()
        result = BlockQuantizer.safe_broadcast_subtract(A_gpu, B_gpu)
        self.assertTrue(torch.equal(result, expected),
                        msg=f"Mismatch for shapes {A.shape} - {B.shape}")

    def test_basic(self):
        A = torch.randn(1024, 64, 1)
        B = torch.randn(1, 256)
        self.compare_with_standard(A, B)

    def test_same_shape(self):
        A = torch.randn(10, 20)
        B = torch.randn(10, 20)
        self.compare_with_standard(A, B)

    def test_scalar_broadcast(self):
        A = torch.randn(32, 32)
        B = torch.tensor(3.14)
        self.compare_with_standard(A, B)

    def test_row_vector_broadcast(self):
        A = torch.randn(64, 128)
        B = torch.randn(128)
        self.compare_with_standard(A, B)

    def test_column_vector_broadcast(self):
        A = torch.randn(64, 128)
        B = torch.randn(64, 1)
        self.compare_with_standard(A, B)

        C = torch.randn(64, 128, 3)
        D = torch.randn(64, 1, 3)
        self.compare_with_standard(C, D)

    def test_high_dimensional(self):
        A = torch.randn(2, 3, 4, 5)
        B = torch.randn(1, 3, 1, 5)
        self.compare_with_standard(A, B)

        C = torch.randn(2, 3, 4, 1)
        D = torch.randn(3, 1, 5)
        self.compare_with_standard(C, D)

        C = torch.randn(20, 30, 40, 50)
        D = torch.randn(1, 30, 1, 50)
        self.compare_with_standard(C, D)

    def test_singleton_dimension(self):
        A = torch.randn(1, 5, 1)
        B = torch.randn(3, 1, 7)
        self.compare_with_standard(A.expand(3, 5, 7), B)

    def test_empty_tensor(self):
        A = torch.randn(0, 10)
        B = torch.randn(10)
        C = torch.randn(1, 10)
        self.compare_with_standard(A, B)
        self.compare_with_standard(B, A)
        self.compare_with_standard(A, C)
        self.compare_with_standard(C, A)

    def test_non_contiguous_tensor(self):
        A = torch.randn(10, 20).t()  # Transposed, non-contiguous
        B = torch.randn(10, 1)
        self.assertRaises(Exception, lambda x: A - B)
        self.assertRaises(Exception, lambda x: BlockQuantizer.safe_broadcast_subtract(A, B, 16))

    def test_mixed_dtypes(self):
        A = torch.randn(16, 16, dtype=torch.float64)
        B = torch.randn(16, dtype=torch.float64)
        self.compare_with_standard(A, B)

    def test_large_tensor(self):
        A = torch.randn(5000, 256)
        B = torch.randn(256)
        self.compare_with_standard(A, B)

if __name__ == '__main__':
    unittest.main()
