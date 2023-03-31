import math
import numpy as np


class ShiftCipher:
    def __init__(self, shift):
        self.shift = shift

    def encrypt(self, plaintext):
        ciphertext = ""
        for char in plaintext:
            if char.isupper():
                ciphertext += chr((ord(char) + self.shift - 65) % 26 + 65)
            elif char.islower():
                ciphertext += chr((ord(char) + self.shift - 97) % 26 + 97)
            else:
                ciphertext += char
        return ciphertext

    def decrypt(self, ciphertext):
        plaintext = ""
        for char in ciphertext:
            if char.isupper():
                plaintext += chr((ord(char) - self.shift - 65) % 26 + 65)
            elif char.islower():
                plaintext += chr((ord(char) - self.shift - 97) % 26 + 97)
            else:
                plaintext += char
        return plaintext


class MonoalphabeticCipher:
    def __init__(self, key):
        new_key = ""
        alphapet = "abcdefghijklmnopqrstuvwxyz"
        for char in key:
            if not char in new_key:
                new_key += char
        for char in alphapet:
            if not char in new_key:
                new_key += char
        self.key = new_key
        print(self.key + "khedr")

    def encrypt(self, plaintext):
        ciphertext = ""
        for char in plaintext:
            if char.isalpha():
                ciphertext += self.key[ord(char) - 97]
            else:
                ciphertext += char
        return ciphertext

    def decrypt(self, ciphertext):
        plaintext = ""
        for char in ciphertext.lower():
            if char.isalpha():
                plaintext += chr(self.key.index(char) + 97)
            else:
                plaintext += char
        return plaintext


class AffineCipher:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

    def encrypt(self, plaintext):
        ciphertext = ""
        for char in plaintext.lower():
            if char in self.alphabet:
                index = self.alphabet.index(char)
                new_index = (self.a * index + self.b) % 26
                ciphertext += self.alphabet[new_index]
            else:
                ciphertext += char
        return ciphertext

    def decrypt(self, ciphertext):
        plaintext = ""
        a_inv = self.mod_inverse(self.a, 26)
        for char in ciphertext.lower():
            if char in self.alphabet:
                index = self.alphabet.index(char)
                new_index = (a_inv * (index - self.b)) % 26
                plaintext += self.alphabet[new_index]
            else:
                plaintext += char
        return plaintext

    def mod_inverse(self, a, m):
        for i in range(1, m):
            if (a * i) % m == 1:
                return i
        return -1


class SubstitutionCipher:
    def __init__(self, key):
        self.key = key.lower()
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

    def encrypt(self, plaintext):
        ciphertext = ""
        for char in plaintext.lower():
            if char in self.alphabet:
                index = self.alphabet.index(char)
                new_char = self.key[index]
                ciphertext += new_char
            else:
                ciphertext += char
        return ciphertext

    def decrypt(self, ciphertext):
        plaintext = ""
        for char in ciphertext.lower():
            if char in self.key:
                index = self.key.index(char)
                new_char = self.alphabet[index]
                plaintext += new_char
            else:
                plaintext += char
        return plaintext


class PlayfairCipher:
    def __init__(self, key):
        self.key = key
        self.key_matrix = self._generate_key_matrix()

    def encrypt(self, plaintext):
        pairs = self._prepare_input(plaintext)
        ciphertext = ""
        for pair in pairs:
            ciphertext += self._encrypt_pair(pair)
        return ciphertext

    def decrypt(self, ciphertext):
        pairs = self._prepare_input(ciphertext)
        plaintext = ""
        for pair in pairs:
            plaintext += self._decrypt_pair(pair)
        return plaintext

    def _generate_key_matrix(self):
        key_matrix = []
        key_chars = []
        for char in self.key:
            if char not in key_chars:
                key_chars.append(char)
        for i in range(ord("a"), ord("z") + 1):
            char = chr(i)
            if char == "j":
                continue
            if char not in key_chars:
                key_chars.append(char)
        for i in range(0, 25, 5):
            row = key_chars[i : i + 5]
            key_matrix.append(row)
        return key_matrix

    def _prepare_input(self, text):
        text = text.lower().replace(" ", "").replace("j", "i")
        if len(text) % 2 == 1:
            text += "x"
        pairs = []
        i = 0
        while i < len(text):
            pair = text[i : i + 2]
            pairs.append(pair)
            i += 2
        return pairs

    def _encrypt_pair(self, pair):
        char1, char2 = pair[0], pair[1]
        row1, col1 = self._find_char(char1)
        row2, col2 = self._find_char(char2)
        if row1 == row2:
            return (
                self.key_matrix[row1][(col1 + 1) % 5]
                + self.key_matrix[row2][(col2 + 1) % 5]
            )
        elif col1 == col2:
            return (
                self.key_matrix[(row1 + 1) % 5][col1]
                + self.key_matrix[(row2 + 1) % 5][col2]
            )
        else:
            return self.key_matrix[row1][col2] + self.key_matrix[row2][col1]

    def _decrypt_pair(self, pair):
        char1, char2 = pair[0], pair[1]
        row1, col1 = self._find_char(char1)
        row2, col2 = self._find_char(char2)
        if row1 == row2:
            return (
                self.key_matrix[row1][(col1 - 1) % 5]
                + self.key_matrix[row2][(col2 - 1) % 5]
            )
        elif col1 == col2:
            return (
                self.key_matrix[(row1 - 1) % 5][col1]
                + self.key_matrix[(row2 - 1) % 5][col2]
            )
        else:
            return self.key_matrix[row1][col2] + self.key_matrix[row2][col1]

    def _find_char(self, char):
        for i in range(5):
            for j in range(5):
                if self.key_matrix[i][j] == char:
                    return i, j


class VigenereCipher:
    def __init__(self, keyword):
        self.keyword = keyword

    def encrypt(self, plaintext):
        ciphertext = ""
        keyword_index = 0
        for char in plaintext:
            if char.isalpha():
                char = char.upper()
                keyword_char = self.keyword[keyword_index].upper()
                keyword_index = (keyword_index + 1) % len(self.keyword)
                shift = ord(keyword_char) - ord("A")
                char = chr(((ord(char) - ord("A") + shift) % 26) + ord("A"))
            ciphertext += char
        return ciphertext

    def decrypt(self, ciphertext):
        plaintext = ""
        keyword_index = 0
        for char in ciphertext:
            if char.isalpha():
                char = char.upper()
                keyword_char = self.keyword[keyword_index].upper()
                keyword_index = (keyword_index + 1) % len(self.keyword)
                shift = ord(keyword_char) - ord("A")
                char = chr(((ord(char) - ord("A") - shift + 26) % 26) + ord("A"))
            plaintext += char
        return plaintext


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


class RailFenceCipher:
    def encrypt(self, plain_text, rails):
        fence = [["\n" for i in range(len(plain_text))] for j in range(rails)]
        direction = -1
        row, col = 0, 0
        for char in plain_text:
            if row == 0 or row == rails - 1:
                direction *= -1
            fence[row][col] = char
            col += 1
            row += direction
        result = []
        for i in range(rails):
            for j in range(len(plain_text)):
                if fence[i][j] != "\n":
                    result.append(fence[i][j])
        return "".join(result)

    def decrypt(self, cipher_text, rails):
        fence = [["\n" for i in range(len(cipher_text))] for j in range(rails)]
        direction = -1
        row, col = 0, 0
        for i in range(len(cipher_text)):
            if row == 0 or row == rails - 1:
                direction *= -1
            fence[row][col] = "*"
            col += 1
            row += direction
        index = 0
        for i in range(rails):
            for j in range(len(cipher_text)):
                if (fence[i][j] == "*") and (index < len(cipher_text)):
                    fence[i][j] = cipher_text[index]
                    index += 1
        result = []
        row, col = 0, 0
        for i in range(len(cipher_text)):
            if row == 0 or row == rails - 1:
                direction *= -1
            if fence[row][col] != "*":
                result.append(fence[row][col])
                col += 1
            row += direction
        return "".join(result)


class RowTranspositionCipher:
    def __init__(self, key):
        self.key = key

    def encrypt(self, plaintext):
        cipher = ""
        k_indx = 0
        msg_len = float(len(plaintext))
        msg_lst = list(plaintext)
        key_lst = sorted(list(self.key))
        col = len(self.key)
        row = int(math.ceil(msg_len / col))
        fill_null = int((row * col) - msg_len)
        msg_lst.extend("_" * fill_null)
        matrix = [msg_lst[i : i + col] for i in range(0, len(msg_lst), col)]
        for _ in range(col):
            curr_idx = self.key.index(key_lst[k_indx])
            cipher += "".join([row[curr_idx] for row in matrix])
            k_indx += 1
        return cipher

    def decrypt(self, ciphertext):
        plaintext = ""
        k_indx = 0
        msg_indx = 0
        msg_len = float(len(ciphertext))
        msg_lst = list(ciphertext)
        col = len(self.key)
        row = int(math.ceil(msg_len / col))
        key_lst = sorted(list(self.key))
        dec_cipher = []
        for _ in range(row):
            dec_cipher += [[None] * col]
        for _ in range(col):
            curr_idx = self.key.index(key_lst[k_indx])
            for j in range(row):
                dec_cipher[j][curr_idx] = msg_lst[msg_indx]
                msg_indx += 1
            k_indx += 1
        try:
            plaintext = "".join(sum(dec_cipher, []))
        except TypeError:
            raise TypeError("This program cannot", "handle repeating words.")
        null_count = plaintext.count("_")
        if null_count > 0:
            return plaintext[:-null_count]
        return plaintext


# Testing Inputs
print("\n----Shift Cipher----")
cipher = ShiftCipher(11)
plaintext = "mohamed mahmoud khedr"
ciphertext = cipher.encrypt(plaintext)
print("ciphertext = ", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext)
print("plaintext = ", decrypted_plaintext)

print("\n###################################\n")

print("----Monoalphabetic Cipher----")
cipher = MonoalphabeticCipher("example")
plaintext = "mohamed mahmoud khedr"
ciphertext = cipher.encrypt(plaintext)
print("ciphertext = ", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext)
print("plaintext = ", decrypted_plaintext)

print("\n###################################\n")

print("----Affine Cipher----")
cipher = AffineCipher(7, 3)
plaintext = "mohamed mahmoud khedr"
ciphertext = cipher.encrypt(plaintext)
print("ciphertext = ", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext)
print("plaintext = ", decrypted_plaintext)

print("\n###################################\n")

print("----Substitution Cipher----")
cipher = SubstitutionCipher("xnyahpogzqwbtsflrcvmuekjdi")
plaintext = "mohamed mahmoud khedr"
ciphertext = cipher.encrypt(plaintext)
print("ciphertext = ", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext)
print("plaintext = ", decrypted_plaintext)


print("\n###################################\n")

print("----Playfair Cipher----")
cipher = PlayfairCipher("monarchy")
plaintext = "mohammed mahmoud khedr"
ciphertext = cipher.encrypt(plaintext)
print("ciphertext = ", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext)
print("plaintext = ", decrypted_plaintext)

print("\n###################################\n")

print("----Vigenere Cipher----")
cipher = VigenereCipher("example")
plaintext = "mohamed mahmoud khedr"
ciphertext = cipher.encrypt(plaintext)
print("ciphertext = ", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext)
print("plaintext = ", decrypted_plaintext)

print("\n###################################\n")

print("----Hill Cipher----")
cipher = HillCipher(np.array([[2, 1], [3, 4]]))
plaintext = "mohammed mahmoud khedr"
ciphertext = cipher.encrypt(plaintext)
print("Ciphertext:", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext)
print("plaintext:", decrypted_plaintext)

print("\n###################################\n")

print("----Rail Fence Cipher----")
cipher = RailFenceCipher()
plaintext = "mohamedmahmoudkhedr"
rails = 3
ciphertext = cipher.encrypt(plaintext, rails)
print("ciphertext = ", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext, rails)
print("plaintext = ", decrypted_plaintext)

print("\n###################################\n")

print("----Row Transposition Cipher----")
rt = RowTranspositionCipher("4312567")
plaintext = "mohamed mahmoud khedr"
ciphertext = rt.encrypt(plaintext)
print("ciphertext = ", ciphertext)
decrypted_plaintext = rt.decrypt(ciphertext)
print("plaintext = ", decrypted_plaintext)

print("\n###################################\n")
