import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        mean = 0
        var = 50000
        sigma = var ** 0.5
        if len(image.shape) == 3:
            row, col, ch = image.shape
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
        else:
            row, col = image.shape
            gauss = np.random.normal(mean, sigma, (row, col))
            gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy.astype(int)
    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        prob = 0.5
        output = image.copy()
        if len(image.shape) == 2:
            black = 0
            white = 255
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(output.shape[:2])
        output[probs < (prob / 2)] = black
        output[probs > 1 - (prob / 2)] = white
        return output


def calculate_PSNR(image1, image2):
    mse = np.sum((image1 - image2) ** 2) / np.prod(image1.shape)
    print(mse)
    return 10 * np.log10((255 ** 2) / mse)


def rth_approximation(img, r):
    (m, n) = img.shape
    A = np.zeros((m, n))

    U, S, Vt = np.linalg.svd(img)

    for i in range(r):
        A += S[i] * np.outer(U[:, i], Vt[i, :])

    return A.astype(int)


img = mpimg.imread('q2_pic.jpg')

R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
img = 0.2989 * R + 0.5870 * G + 0.1140 * B

noisy_sp = noisy("s&p", img)
noisy_gauss = noisy("gauss", img)


R = [3, 5, 10, 50, 150, 250]
n = int(np.sqrt(len(R)))+1

PSNR = []

for i in range(len(R)):
    r = R[i]
    plt.subplot(n, int(np.ceil(len(R)/n)), i + 1)
    plt.imshow(rth_approximation(img, r), cmap='gray')
    img_k = rth_approximation(img, r)
    PSNR.append(calculate_PSNR(img, img_k))
    plt.title('k = ' + str(r))


plt.suptitle("SVD approximation")
plt.axis = "off"
plt.savefig('SVD approximation.png')
plt.show()
plt.close()

plt.plot(R, PSNR)
plt.title("PSNR approximation.png")
plt.grid()
plt.savefig('PSNR approximation.png')
plt.show()
plt.close()

PSNR_sp = []
PSNR_g = []

R = [5, 10, 20, 30, 40, 50]

for i in range(len(R)):
    r = R[i]
    plt.subplot(n, int(np.ceil(len(R)/n)), i + 1)
    denoised_img = rth_approximation(noisy_sp, r)
    PSNR_sp.append(calculate_PSNR(img, denoised_img))
    plt.imshow(denoised_img, cmap='gray')
    plt.title('k = ' + str(r))

plt.suptitle("denoised salt and pepper")
plt.axis = "off"
plt.savefig('denoised salt and pepper.png')
plt.show()
plt.close()


for i in range(len(R)):
    r = R[i]
    plt.subplot(n, int(np.ceil(len(R)/n)), i + 1)
    denoised_img = rth_approximation(noisy_gauss, r)
    PSNR_g.append(calculate_PSNR(img, denoised_img))
    plt.imshow(denoised_img, cmap='gray')
    plt.title('k = ' + str(r))

plt.suptitle("denoised gauss")
plt.axis = "off"
plt.savefig('denoised gauss.png')
plt.show()
plt.close()


plt.plot(R, PSNR_sp)
plt.title("PSNR salt and pepper.png")
plt.grid()
plt.savefig('PSNR salt and pepper.png')
plt.show()
plt.close()

plt.plot(R, PSNR_g)
plt.title("PSNR gauss.png")
plt.grid()
plt.savefig('PSNR gauss.png')
plt.show()
plt.close()

if __name__ == "__main__":
    None
