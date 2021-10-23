import cv2 as cv
import numpy as np

i = 1

bruh = np.zeros(1003)

while i < 47:
    img = cv.imread(f"./clothing-co-parsing/cropped/{i:04d}.jpg")
    while True:
        cv.imshow("fuck", img)
        poll = cv.waitKey(10)
        if(poll == ord("y")):
            print(f"{i:04d} is valid.")
            break
        if(poll == ord("n")):
            print(f"{i:04d} is not valid.")
            bruh[i-1] = 1
            break
        if(poll == ord("q")):
            exit(0)
        if(poll == ord("b")):
            i -=2
            break

    i += 1

backup = """0046 is valid.
0047 is valid.
0048 is valid.
0049 is not valid.
0050 is valid.
0050 is not valid.
0051 is valid.
0052 is valid.
0053 is valid.
0054 is not valid.
0055 is valid.
0056 is valid.
0056 is not valid.
0057 is not valid.
0058 is valid.
0059 is valid.
0060 is valid.
0061 is valid.
0062 is valid.
0063 is valid.
0064 is valid.
0065 is valid.
0066 is valid.
0067 is valid.
0068 is valid.
0069 is valid.
0070 is valid.
0071 is valid.
0072 is valid.
0073 is valid.
0074 is valid.
0075 is not valid.
0076 is valid.
0077 is not valid.
0078 is valid.
0079 is valid.
0080 is valid.
0081 is valid.
0082 is valid.
0083 is valid.
0084 is valid.
0085 is not valid.
0086 is valid.
0087 is valid.
0088 is valid.
0089 is valid.
0090 is valid.
0091 is valid.
0092 is valid.
0093 is valid.
0094 is valid.
0095 is valid.
0096 is not valid.
0097 is valid.
0098 is valid.
0099 is valid.
0100 is valid.
0101 is valid.
0102 is valid.
0103 is valid.
0104 is not valid.
0105 is valid.
0106 is valid.
0107 is valid.
0108 is valid.
0109 is not valid.
0110 is valid.
0111 is valid.
0112 is valid.
0113 is valid.
0114 is valid.
0115 is valid.
0116 is valid.
0117 is valid.
0118 is not valid.
0119 is not valid.
0120 is valid.
0121 is valid.
0122 is valid.
0123 is valid.
0124 is valid.
0125 is valid.
0126 is valid.
0127 is valid.
0128 is valid.
0129 is valid.
0130 is valid.
0131 is valid.
0132 is valid.
0133 is valid.
0134 is valid.
0135 is valid.
0136 is valid.
0137 is valid.
0138 is valid.
0139 is valid.
0140 is valid.
0141 is valid.
0142 is valid.
0143 is valid.
0144 is valid.
0145 is valid.
0146 is valid.
0147 is valid.
0148 is valid.
0149 is valid.
0149 is not valid.
0150 is valid.
0151 is valid.
0152 is valid.
0151 is valid.
0152 is not valid.
0152 is not valid.
0153 is valid.
0154 is valid.
0155 is not valid.
0156 is not valid.
0157 is not valid.
0158 is valid.
0159 is valid.
0160 is valid.
0161 is valid.
0162 is valid.
0163 is valid.
0163 is not valid.
0164 is valid.
0165 is valid.
0166 is valid.
0167 is valid.
0168 is valid.
0169 is valid.
0170 is valid.
0171 is valid.
0172 is valid.
0173 is not valid.
0174 is valid.
0175 is valid.
0176 is valid.
0177 is valid.
0178 is valid.
0179 is valid.
0180 is valid.
0181 is valid.
0182 is valid.
0183 is not valid.
0184 is valid.
0185 is valid.
0186 is valid.
0187 is valid.
0188 is valid.
0189 is valid.
0190 is valid.
0191 is valid.
0192 is valid.
0193 is valid.
0194 is valid.
0195 is valid.
0196 is valid.
0196 is not valid.
0197 is valid.
0198 is valid.
0199 is valid.
0200 is valid.
0201 is valid.
0202 is valid.
0203 is valid.
0204 is valid.
0205 is valid.
0206 is valid.
0207 is valid.
0207 is not valid.
0208 is valid.
0209 is valid.
0210 is valid.
0211 is valid.
0209 is not valid.
0210 is valid.
0211 is valid.
0212 is valid.
0213 is valid.
0214 is valid.
0215 is valid.
0216 is valid.
0217 is valid.
0218 is valid.
0219 is not valid.
0220 is valid.
0221 is valid.
0222 is valid.
0223 is valid.
0224 is valid.
0225 is valid.
0226 is valid.
0227 is valid.
0228 is valid.
0229 is valid.
0230 is not valid.
0229 is valid.
0230 is not valid.
0231 is valid.
0232 is valid.
0233 is valid.
0234 is valid.
0235 is valid.
0236 is valid.
0237 is valid.
0238 is not valid.
0239 is valid.
0240 is valid.
0241 is valid.
0242 is valid.
0243 is not valid.
0244 is valid.
0245 is valid.
0246 is valid.
0247 is valid.
0248 is valid.
0249 is not valid.
0250 is not valid.
0245 is valid.
0246 is valid.
0247 is valid.
0248 is not valid.
0249 is valid.
0250 is not valid.
0251 is not valid.
0252 is valid.
0253 is not valid.
0254 is valid.
0255 is not valid.
0256 is valid.
0257 is not valid.
0258 is valid.
0257 is valid.
0258 is valid.
0259 is not valid.
0260 is valid.
0261 is valid.
0262 is valid.
0263 is valid.
0264 is valid.
0265 is valid.
0266 is valid.
0267 is valid.
0268 is not valid.
0269 is valid.
0270 is valid.
0271 is valid.
0272 is valid.
0273 is valid.
0274 is not valid.
0275 is valid.
0276 is valid.
0277 is valid.
0278 is valid.
0279 is valid.
0280 is valid.
0281 is valid.
0282 is not valid.
0283 is valid.
0284 is not valid.
0285 is valid.
0286 is valid.
0287 is valid.
0288 is valid.
0289 is valid.
0290 is valid.
0291 is valid.
0292 is valid.
0293 is not valid.
0294 is valid.
0295 is valid.
0296 is valid.
0297 is valid.
0298 is not valid.
0299 is valid.
0299 is not valid.
0300 is valid.
0301 is valid.
0302 is valid.
0302 is valid.
0303 is not valid.
0304 is valid.
0305 is not valid.
0306 is valid.
0307 is valid.
0308 is valid.
0309 is valid.
0310 is valid.
0310 is valid.
0311 is valid.
0312 is valid.
0313 is valid.
0314 is valid.
0315 is valid.
0316 is valid.
0317 is valid.
0318 is valid.
0319 is valid.
0320 is valid.
0321 is valid.
0322 is valid.
0323 is valid.
0324 is valid.
0325 is valid.
0326 is valid.
0327 is valid.
0328 is valid.
0329 is valid.
0330 is valid.
0331 is valid.
0332 is valid.
0333 is valid.
0334 is not valid.
0335 is valid.
0336 is valid.
0337 is valid.
0338 is valid.
0339 is not valid.
0340 is valid.
0341 is valid.
0342 is valid.
0343 is valid.
0344 is valid.
0345 is valid.
0346 is valid.
0347 is valid.
0348 is not valid.
0349 is valid.
0349 is not valid.
0350 is valid.
0351 is valid.
0352 is valid.
0353 is not valid.
0354 is valid.
0355 is valid.
0355 is not valid.
0356 is not valid.
0357 is valid.
0358 is valid.
0359 is not valid.
0360 is valid.
0361 is valid.
0362 is valid.
0363 is not valid.
0364 is valid.
0365 is valid.
0366 is valid.
0367 is valid.
0368 is not valid.
0369 is not valid.
0370 is not valid.
0371 is not valid.
0372 is valid.
0373 is not valid.
0374 is valid.
0375 is not valid.
0376 is valid.
0377 is valid.
0378 is not valid.
0379 is valid.
0380 is valid.
0381 is valid.
0382 is valid.
0383 is valid.
0384 is valid.
0385 is not valid.
0386 is valid.
0387 is not valid.
0388 is valid.
0389 is valid.
0390 is not valid.
0391 is valid.
0392 is valid.
0393 is valid.
0394 is valid.
0395 is valid.
0396 is not valid.
0396 is valid.
0397 is valid.
0398 is valid.
0399 is not valid.
0400 is valid.
0401 is valid.
0402 is valid.
0403 is valid.
0404 is valid.
0405 is valid.
0406 is valid.
0407 is valid.
0408 is valid.
0409 is valid.
0410 is valid.
0411 is valid.
0412 is valid.
0413 is valid.
0414 is valid.
0415 is valid.
0416 is valid.
0417 is valid.
0418 is valid.
0419 is valid.
0420 is not valid.
0421 is valid.
0422 is valid.
0423 is valid.
0424 is valid.
0425 is not valid.
0426 is valid.
0427 is valid.
0428 is valid.
0429 is valid.
0430 is not valid.
0431 is valid.
0432 is valid.
0433 is not valid.
0434 is valid.
0435 is valid.
0436 is not valid.
0437 is valid.
0438 is valid.
0439 is not valid.
0440 is valid.
0441 is valid.
0442 is valid.
0443 is valid.
0444 is valid.
0445 is valid.
0446 is valid.
0447 is valid.
0448 is valid.
0449 is valid.
0450 is not valid.
0451 is valid.
0452 is valid.
0453 is valid.
0453 is not valid.
0454 is valid.
0455 is not valid.
0456 is not valid.
0457 is valid.
0458 is valid.
0459 is valid.
0460 is valid.
0461 is valid.
0462 is valid.
0463 is valid.
0464 is valid.
0465 is valid.
0466 is not valid.
0467 is valid.
0467 is not valid.
0468 is valid.
0469 is valid.
0470 is valid.
0471 is valid.
0472 is not valid.
0473 is valid.
0473 is not valid.
0474 is valid.
0475 is valid.
0475 is not valid.
0476 is not valid.
0477 is valid.
0478 is valid.
0479 is not valid.
0480 is valid.
0481 is valid.
0482 is valid.
0483 is valid.
0484 is valid.
0485 is valid.
0486 is valid.
0487 is valid.
0488 is valid.
0489 is valid.
0490 is not valid.
0491 is valid.
0489 is valid.
0490 is not valid.
0491 is not valid.
0492 is valid.
0493 is valid.
0494 is valid.
0495 is valid.
0496 is not valid.
0497 is valid.
0498 is valid.
0499 is not valid.
0500 is valid.
0501 is not valid.
0502 is valid.
0503 is valid.
0504 is valid.
0505 is valid.
0506 is valid.
0507 is valid.
0508 is not valid.
0509 is valid.
0510 is not valid.
0511 is valid.
0512 is valid.
0513 is valid.
0514 is valid.
0515 is valid.
0516 is not valid.
0517 is valid.
0518 is valid.
0519 is valid.
0520 is valid.
0521 is not valid.
0522 is valid.
0523 is valid.
0524 is valid.
0525 is valid.
0526 is valid.
0527 is valid.
0528 is valid.
0529 is valid.
0530 is valid.
0531 is valid.
0532 is valid.
0533 is not valid.
0534 is valid.
0535 is valid.
0534 is valid.
0535 is not valid.
0536 is not valid.
0537 is not valid.
0538 is valid.
0539 is valid.
0540 is not valid.
0541 is valid.
0542 is valid.
0543 is valid.
0544 is valid.
0545 is valid.
0546 is valid.
0547 is valid.
0548 is not valid.
0549 is valid.
0550 is not valid.
0551 is not valid.
0552 is valid.
0553 is valid.
0554 is valid.
0555 is not valid.
0556 is valid.
0557 is valid.
0558 is valid.
0559 is valid.
0560 is valid.
0561 is valid.
0562 is valid.
0563 is valid.
0564 is valid.
0565 is valid.
0566 is not valid.
0567 is valid.
0567 is not valid.
0568 is valid.
0569 is not valid.
0570 is valid.
0571 is valid.
0572 is valid.
0573 is valid.
0574 is valid.
0575 is valid.
0576 is valid.
0577 is valid.
0578 is valid.
0579 is valid.
0580 is not valid.
0581 is valid.
0582 is not valid.
0583 is valid.
0584 is valid.
0585 is valid.
0586 is valid.
0587 is valid.
0588 is valid.
0589 is valid.
0590 is valid.
0591 is not valid.
0592 is valid.
0593 is not valid.
0594 is valid.
0595 is valid.
0596 is not valid.
0597 is valid.
0597 is not valid.
0598 is valid.
0599 is valid.
0600 is valid.
0601 is valid.
0602 is valid.
0603 is not valid.
0604 is valid.
0605 is valid.
0606 is valid.
0607 is valid.
0608 is not valid.
0609 is not valid.
0610 is valid.
0611 is valid.
0612 is valid.
0613 is valid.
0614 is valid.
0615 is valid.
0616 is valid.
0617 is valid.
0618 is valid.
0619 is valid.
0620 is valid.
0621 is valid.
0621 is not valid.
0622 is valid.
0623 is not valid.
0623 is valid.
0624 is valid.
0625 is valid.
0626 is not valid.
0627 is valid.
0628 is not valid.
0629 is not valid.
0630 is valid.
0631 is valid.
0632 is valid.
0633 is not valid.
0634 is valid.
0635 is valid.
0636 is valid.
0637 is valid.
0638 is valid.
0639 is valid.
0640 is valid.
0641 is valid.
0642 is valid.
0643 is valid.
0644 is valid.
0645 is valid.
0646 is valid.
0647 is valid.
0648 is valid.
0649 is valid.
0650 is valid.
0650 is not valid.
0651 is valid.
0652 is valid.
0653 is valid.
0654 is valid.
0655 is valid.
0656 is valid.
0657 is valid.
0658 is valid.
0659 is valid.
0660 is valid.
0661 is not valid.
0662 is valid.
0663 is valid.
0664 is valid.
0665 is valid.
0666 is valid.
0667 is valid.
0668 is valid.
0669 is valid.
0670 is valid.
0671 is valid.
0672 is valid.
0673 is valid.
0674 is valid.
0675 is valid.
0676 is valid.
0677 is valid.
0678 is valid.
0679 is valid.
0680 is valid.
0681 is valid.
0682 is valid.
0683 is valid.
0684 is not valid.
0685 is valid.
0686 is valid.
0687 is not valid.
0688 is valid.
0689 is valid.
0690 is valid.
0691 is valid.
0692 is valid.
0693 is valid.
0694 is valid.
0695 is valid.
0696 is valid.
0697 is valid.
0698 is valid.
0699 is valid.
0700 is valid.
0701 is valid.
0702 is valid.
0703 is valid.
0704 is valid.
0705 is valid.
0706 is valid.
0707 is valid.
0708 is valid.
0709 is valid.
0710 is valid.
0711 is valid.
0712 is valid.
0713 is valid.
0714 is valid.
0715 is valid.
0716 is valid.
0717 is valid.
0718 is valid.
0719 is valid.
0720 is valid.
0720 is not valid.
0721 is valid.
0722 is valid.
0721 is valid.
0722 is not valid.
0723 is not valid.
0724 is valid.
0725 is valid.
0726 is not valid.
0727 is valid.
0728 is valid.
0729 is not valid.
0730 is valid.
0730 is not valid.
0731 is valid.
0732 is valid.
0733 is valid.
0734 is valid.
0735 is valid.
0736 is not valid.
0737 is not valid.
0738 is valid.
0739 is valid.
0740 is valid.
0740 is not valid.
0741 is valid.
0742 is not valid.
0743 is valid.
0744 is valid.
0745 is valid.
0746 is valid.
0747 is valid.
0748 is valid.
0749 is valid.
0750 is valid.
0751 is valid.
0752 is not valid.
0753 is valid.
0754 is valid.
0755 is valid.
0756 is valid.
0757 is valid.
0758 is valid.
0759 is valid.
0760 is valid.
0761 is not valid.
0762 is valid.
0763 is valid.
0764 is valid.
0765 is valid.
0765 is not valid.
0766 is valid.
0767 is valid.
0768 is valid.
0769 is valid.
0770 is valid.
0771 is valid.
0772 is valid.
0773 is valid.
0774 is valid.
0775 is valid.
0776 is valid.
0777 is not valid.
0778 is valid.
0779 is valid.
0780 is valid.
0781 is valid.
0782 is valid.
0783 is valid.
0784 is valid.
0785 is valid.
0786 is valid.
0785 is valid.
0786 is not valid.
0787 is valid.
0788 is valid.
0789 is valid.
0790 is valid.
0791 is not valid.
0792 is not valid.
0793 is valid.
0794 is not valid.
0795 is valid.
0796 is valid.
0797 is valid.
0798 is valid.
0799 is valid.
0800 is valid.
0801 is valid.
0802 is valid.
0803 is valid.
0804 is not valid.
0805 is valid.
0806 is not valid.
0807 is not valid.
0808 is valid.
0809 is valid.
0810 is valid.
0811 is not valid.
0812 is valid.
0813 is valid.
0814 is valid.
0815 is valid.
0816 is valid.
0817 is not valid.
0818 is valid.
0819 is valid.
0820 is valid.
0821 is not valid.
0822 is valid.
0823 is valid.
0824 is not valid.
0824 is valid.
0825 is valid.
0826 is valid.
0827 is valid.
0828 is not valid.
0829 is valid.
0830 is valid.
0831 is valid.
0832 is valid.
0833 is valid.
0834 is not valid.
0835 is valid.
0836 is valid.
0837 is not valid.
0838 is valid.
0839 is valid.
0840 is not valid.
0841 is valid.
0842 is not valid.
0843 is valid.
0844 is valid.
0845 is not valid.
0846 is valid.
0847 is valid.
0848 is not valid.
0849 is valid.
0850 is not valid.
0851 is valid.
0852 is valid.
0853 is not valid.
0854 is not valid.
0855 is not valid.
0856 is valid.
0857 is valid.
0858 is valid.
0859 is valid.
0860 is valid.
0861 is valid.
0862 is not valid.
0863 is valid.
0864 is valid.
0865 is not valid.
0866 is not valid.
0867 is valid.
0868 is not valid.
0869 is valid.
0870 is valid.
0871 is valid.
0872 is not valid.
0873 is valid.
0874 is valid.
0875 is valid.
0876 is valid.
0877 is valid.
0878 is valid.
0879 is valid.
0880 is valid.
0881 is valid.
0882 is not valid.
0883 is valid.
0884 is valid.
0885 is valid.
0886 is valid.
0887 is valid.
0888 is valid.
0889 is not valid.
0890 is valid.
0891 is not valid.
0891 is valid.
0892 is valid.
0893 is valid.
0894 is valid.
0895 is not valid.
0896 is valid.
0897 is valid.
0898 is valid.
0899 is valid.
0900 is valid.
0901 is not valid.
0902 is valid.
0903 is valid.
0904 is not valid.
0905 is valid.
0905 is valid.
0906 is not valid.
0907 is valid.
0908 is not valid.
0909 is valid.
0910 is valid.
0911 is valid.
0912 is valid.
0913 is valid.
0914 is valid.
0915 is valid.
0916 is valid.
0917 is valid.
0918 is valid.
0919 is valid.
0920 is not valid.
0921 is valid.
0922 is valid.
0923 is valid.
0924 is valid.
0925 is not valid.
0926 is not valid.
0927 is valid.
0928 is valid.
0929 is valid.
0930 is not valid.
0931 is valid.
0932 is valid.
0933 is valid.
0934 is not valid.
0935 is valid.
0936 is valid.
0937 is not valid.
0938 is not valid.
0939 is valid.
0940 is valid.
0941 is valid.
0942 is valid.
0943 is valid.
0944 is valid.
0945 is valid.
0946 is valid.
0947 is valid.
0948 is valid.
0949 is valid.
0950 is not valid.
0951 is valid.
0952 is valid.
0953 is valid.
0954 is not valid.
0955 is valid.
0956 is not valid.
0957 is not valid.
0958 is valid.
0959 is valid.
0960 is valid.
0961 is valid.
0962 is valid.
0963 is valid.
0964 is valid.
0965 is valid.
0966 is valid.
0967 is not valid.
0968 is valid.
0969 is not valid.
0970 is valid.
0971 is valid.
0972 is not valid.
0973 is valid.
0974 is valid.
0975 is valid.
0976 is not valid.
0977 is valid.
0978 is not valid.
0979 is valid.
0980 is valid.
0981 is valid.
0982 is valid.
0983 is valid.
0984 is valid.
0985 is not valid.
0986 is valid.
0987 is not valid.
0988 is valid.
0989 is not valid.
0989 is valid.
0990 is valid.
0991 is valid.
0992 is valid.
0993 is not valid.
0994 is valid.
0995 is valid.
0996 is valid.
0997 is valid.
0998 is valid.
0999 is not valid.
1000 is valid.
1001 is valid.
1002 is not valid."""



for line in backup.split("\n"):
    a = int(line[0:4])
    b = line[8:11] == "not"
    print(line[0:4], end=" ")
    if(b):
        print("b")
        bruh[a] = 1
    else:
        print("nv")
np.savetxt("invalid.csv", bruh, fmt="%d", delimiter=", ", newline=", ")

