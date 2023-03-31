import math


class RowTransposition:
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


rt = RowTransposition("4312567")
plaintext = "mohamed mahmoud khedr"
ciphertext = rt.encrypt(plaintext)
print("ciphertext = ", ciphertext)
decrypted_plaintext = rt.decrypt(ciphertext)
print("plaintext = ", decrypted_plaintext)
