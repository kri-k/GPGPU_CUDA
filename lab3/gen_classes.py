import sys
from PIL import Image, ImageDraw


CLS = [
    [(0, 0, 0)],
    [(255, 0, 0), (128, 0, 0)],
    [(0, 255, 0), (0, 128, 0)],
    [(0, 0, 255), (0, 0, 128)],
    [(255, 255, 255), (128, 128, 128)],
    [(0, 255, 255), (0, 128, 128)],
    [(255, 0, 255), (128, 0, 128)],
    [(255, 255, 0), (128, 128, 0)]
]


def gen_cls(in_img_path, out_txt_path):
    im = Image.open(in_img_path)
    w, h = im.size

    pix = im.convert('RGB').load()

    cls = [[] for _ in CLS]

    for i in range(w):
        for j in range(h):
            l = [x for x in range(len(CLS)) if pix[i, j] in CLS[x]]
            if len(l) == 0:
                  continue
            cls[l[0]].append((i, j))

    fout = open(out_txt_path, 'w')
    print('in.data\nout.data', file=fout)
    print(len(CLS), file=fout)
    for c in cls:
        print(len(c), end=' ', file=fout)
        for p in c:
            print(*p, end=' ', file=fout)
        print(file=fout)
    fout.close()


def gen_img(in_img_path):
    out_img_path = 'classified.png'
    im = Image.open(in_img_path)
    w, h = im.size

    draw = ImageDraw.Draw(im)

    pix = im.convert('RGBA').load()

    for i in range(w):
        for j in range(h):
            a = pix[i, j][3]
            if a >= len(CLS):
                draw.point((i, j), (0, 0, 0))
            else:
                draw.point((i, j), CLS[a][0])

    im.save(out_img_path)


def main():
    if len(sys.argv) == 2:
        gen_img(sys.argv[1])
    else:
        gen_cls(*sys.argv[1:3])


if __name__ == '__main__':
	main()
