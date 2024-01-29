import numpy as np
import time
import pickle


def crc_6_itu(code, poly=np.array([1, 0, 0, 0, 0, 1, 1])):
    def squeeze(a):
        k = 0
        while k < len(a):
            if a[k] == 1:
                break
            else:
                k += 1
        return a[k:]

    length = len(poly)
    code = squeeze(code)
    while len(code) >= length:
        remainder = np.bitwise_xor(code[:length], poly)
        remainder = squeeze(remainder)
        code = np.hstack((remainder, code[length:]))
        code = squeeze(code)
    remainder = code
    return remainder


def hamming(raw_code, codes):
    code_len = np.prod(raw_code.shape)
    if len(codes) == 0:
        return np.array([code_len])
    rotated_codes = np.empty((0, 16))
    for i in range(4):
        rotated_code = np.rot90(raw_code, i)
        rotated_code = rotated_code.ravel()
        rotated_codes = np.vstack((rotated_codes, rotated_code))
    hamming_dst = np.min(code_len - np.sum(rotated_codes[:, None] == codes[None], axis=-1), axis=0)
    return hamming_dst


def check(code, id_len=10, poly=[1, 0, 0, 0, 0, 1, 1]):
    res = []
    ori = 1
    for i in range(1, 5):
        ori_code = np.rot90(code, i)
        ori_code = ori_code.ravel()
        remainder = crc_6_itu(ori_code, poly)
        if len(remainder) == 0:
            # num = np.sum(ori_code[:id_len] * np.power(2, np.arange((id_len - 1), -1, -1)))
            num = int(''.join([str(b) for b in ori_code[:id_len]]), base=2)
            res.append(1)
            ori = i
        else:
            res.append(0)
    if np.sum(res) == 1:
        return num, ori
    else:
        return -1, ori


def num2code(num, id_len=10, check_len=6, poly=[1, 0, 0, 0, 0, 1, 1]):
    bin_num = bin(num)
    bin_num = bin_num[2:]
    identifier = np.zeros(id_len, dtype=np.int32)
    bin_num = np.array([int(bit) for bit in bin_num])
    identifier[-len(bin_num):] = bin_num
    new_identifier = np.hstack((identifier, np.zeros(check_len, dtype=np.int32)))
    remainder = crc_6_itu(new_identifier, poly)
    checksum = np.zeros(check_len, dtype=np.int32)
    if len(remainder) > 0:
        checksum[-len(remainder):] = remainder
    code = np.hstack((identifier, checksum)).reshape(4, 4)
    return code


def constraint(id_len=10, check_len=6, poly=[1, 0, 0, 0, 0, 1, 1]):
    nums = []
    for i in range(pow(2, id_len)):
        code = num2code(i, id_len, check_len, poly)
        if 4 <= np.sum(code) <= 12 and check(code, id_len, poly)[0] > -1:
            nums.append(i)
    return nums


def greedy_search(id_len=10, check_len=6, poly=[1, 0, 0, 0, 0, 1, 1], hamming_distance=3):
    code_books = np.empty((0, 16))
    code_len = id_len + check_len
    nums = constraint(id_len, check_len, poly)
    distance = np.empty(0)
    for num in nums:
        code = num2code(num, id_len, check_len, poly)
        code_books = np.vstack((code_books, code.ravel()))
    for i, num in enumerate(nums):
        code = num2code(num, id_len, check_len, poly)
        new_code_books = np.vstack((code_books[:i], code_books[i + 1:]))
        distance = np.append(distance, np.sum(hamming(code, new_code_books) >= hamming_distance))
    orders = np.argsort(distance)[::-1]
    codes = np.empty((0, 16))
    res = []
    for index in orders:
        code = num2code(nums[index], id_len, check_len, poly)
        ham = hamming(code, codes)
        if np.all(ham >= hamming_distance):
            res.append(nums[index])
            codes = np.vstack((codes, code.ravel()))
    return res


def sequential_search(id_len=10, check_len=6, poly=[1, 0, 0, 0, 0, 1, 1], hamming_distance=3):
    valid_codes = np.empty((0, 16))
    valid_nums = []
    nums = constraint(id_len, check_len, poly)
    for num in nums:
        code = num2code(num, id_len, check_len, poly)
        ham = hamming(code, valid_codes)
        if np.all(ham >= hamming_distance):
            valid_codes = np.vstack((valid_codes, code.ravel()))
            valid_nums.append(num)
    return valid_nums


def max_capacity(id_len=10, check_len=6, poly=[1, 0, 0, 0, 0, 1, 1], hamming_distance=3):

    def put_down(set1, set2):
        if len(set1) > 0:
            set2_ = set2[mask[set1[-1], set2] == 1]
        else:
            set2_ = set2
        if len(set2_) == 0:
            results.append(set1)
            return
        for ind, number in enumerate(set2_):
            set1_ = set1.copy()
            set1_.append(number)
            set2_ = set2_[ind+1:]
            put_down(set1_, set2_)

    results = []
    nums = constraint(id_len, check_len, poly)
    n = len(nums)
    mask = np.zeros((n, n), np.int32)
    code_set = []
    for num in nums:
        code_ = num2code(num, id_len, check_len, poly)
        code_set.append(code_)
    for i in range(n - 1):
        mask[i, i + 1:] = hamming(code_set[i], np.array(code_set[i+1:]).reshape(-1, 16)) >= hamming_distance
    index_set1 = []
    index_set2 = np.arange(n)
    put_down(index_set1, index_set2)
    return results


if __name__ == '__main__':
    valid_tags = sequential_search(hamming_distance=4)
    print(valid_tags)
    print(len(valid_tags))
    np.save('robustCodeList.npy', np.array(valid_tags))
    # example_sets = {}
    # for k in range(2, 6):
    #     print(k, end=' ')
    #     print(len(sequential_search(hamming_distance=k)), end=' ')
    #     print(len(greedy_search(hamming_distance=k)))
    # #     t0 = time.time()
    # #     result = max_capacity(hamming_distance=k)
    # #     example_sets[k] = result
    # #     print(max(map(len, result)), time.time() - t0)
    # # with open('sets.pkl', 'wb') as f:
    # #     pickle.dump(example_sets, f)

