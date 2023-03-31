import numpy as np


class HillCipher:
    def __init__(self, key):
        self.key = key
        self.key_dim = key.shape[0]

    def encrypt(self, plaintext):
        plaintext = plaintext.upper().replace(" ", "")
        while len(plaintext) % self.key_dim != 0:
            plaintext += "X"
        key_matrix = self.key
        ciphertext = ""
        for i in range(0, len(plaintext), self.key_dim):
            block = np.array(
                [ord(c) - ord("A") for c in plaintext[i : i + self.key_dim]]
            )
            block = block.reshape((self.key_dim, 1))
            encrypted_block = np.dot(key_matrix, block) % 26
            ciphertext += "".join(
                [chr(c + ord("A")) for c in encrypted_block.flatten()]
            )
        return ciphertext

    def decrypt(self, ciphertext):
        ciphertext = ciphertext.upper().replace(" ", "")
        key_matrix = self.key
        det = np.linalg.det(key_matrix)
        det_inv = self.modular_inverse(int(det), 26)
        adj_matrix = np.round(
            det_inv * np.linalg.det(key_matrix) * np.linalg.inv(key_matrix)
        ).astype(int)
        adj_matrix = adj_matrix % 26
        inverse_matrix = np.zeros((self.key_dim, self.key_dim), dtype=int)
        for i in range(self.key_dim):
            for j in range(self.key_dim):
                inverse_matrix[i][j] = adj_matrix[i][j]
        plaintext = ""
        for i in range(0, len(ciphertext), self.key_dim):
            block = np.array(
                [ord(c) - ord("A") for c in ciphertext[i : i + self.key_dim]]
            )
            block = block.reshape((self.key_dim, 1))
            decrypted_block = np.dot(inverse_matrix, block) % 26
            plaintext += "".join([chr(c + ord("A")) for c in decrypted_block.flatten()])
        return plaintext

    def modular_inverse(self, a, m):
        t, newt = 0, 1
        r, newr = m, a
        while newr != 0:
            quotient = r // newr
            t, newt = newt, t - quotient * newt
            r, newr = newr, r - quotient * newr
        if r > 1:
            raise Exception("a is not invertible")
        if t < 0:
            t += m
        return t


key = np.array([[2, 1], [3, 4]])
cipher = HillCipher(key)
plaintext = "mohammed mahmoud khedr"
print("Plaintext:", plaintext)
ciphertext = cipher.encrypt(plaintext)
print("Ciphertext:", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext)
print("Decrypted plaintext:", decrypted_plaintext)
