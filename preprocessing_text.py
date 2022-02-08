import numpy as np
import easydict
import os
import cv2

FLAGS = easydict.EasyDict({"img_size": 256,
                           
                           "cat_img_path": "E:/[1]DB/Cat_and_dog/Cat/",

                           "dog_img_path": "E:/[1]DB/Cat_and_dog/Dog/",
                           
                           "lab_path": "E:/etc_label/label.txt",
                           
                           "batch_size": 2})
# Cat = 0, Dog = 1

def main():
  
    cat_tr_txt = open("E:/[1]DB/Cat_and_dog/cat_train.txt", "w")
    cat_val_txt = open("E:/[1]DB/Cat_and_dog/cat_val.txt", "w")
    cat_te_txt = open("E:/[1]DB/Cat_and_dog/cat_test.txt", "w")

    dog_tr_txt = open("E:/[1]DB/Cat_and_dog/dog_train.txt", "w")
    dog_val_txt = open("E:/[1]DB/Cat_and_dog/dog_val.txt", "w")
    dog_te_txt = open("E:/[1]DB/Cat_and_dog/dog_test.txt", "w")

    cat_list = os.listdir(FLAGS.cat_img_path)
    cat_list_buf = []
    cat_label_buf = []
    for i in range(len(cat_list)):
        if cat_list[i] != "Thumbs.db":
            cat_list_buf.append(cat_list[i])
            cat_label_buf.append(0)

    dog_list = os.listdir(FLAGS.dog_img_path)
    dog_list_buf = []
    dog_label_buf = []
    for i in range(len(dog_list)):
        if dog_list[i] != "Thumbs.db":
            dog_list_buf.append(dog_list[i])
            dog_label_buf.append(1)

    cat_list_buf = [FLAGS.cat_img_path + data for data in cat_list_buf]
    dog_list_buf = [FLAGS.dog_img_path + data for data in dog_list_buf]
    cat_train_img_buf = np.array(cat_list_buf[0:500])
    dog_train_img_buf = np.array(dog_list_buf[0:500])

    cat_val_img_buf = np.array(cat_list_buf[500:600])
    dog_val_img_buf = np.array(dog_list_buf[500:600])

    cat_test_img_buf = np.array(cat_list_buf[600:800])
    dog_test_img_buf = np.array(dog_list_buf[600:800])

    for tr in range(len(cat_train_img_buf)):
        cat_img = cv2.imread(cat_train_img_buf[tr], 1)
        dog_img = cv2.imread(dog_train_img_buf[tr], 1)

        if cat_img is None:
            print("Cat image = {}".format(cat_train_img_buf[tr].split("/")[-1]))
        else:
            cat_tr_txt.write("Cat_" + cat_train_img_buf[tr].split("/")[-1])
            cat_tr_txt.write(" ")
            cat_tr_txt.write("0")
            cat_tr_txt.write("\n")
            cat_tr_txt.flush()
            cv2.imwrite("E:/[1]DB/Cat_and_dog/Cat_Dog_500_train/" + "Cat_" + cat_train_img_buf[tr].split("/")[-1], cat_img)

        if dog_img is None:
            print("Dog image = {}".format(dog_train_img_buf[tr].split("/")[-1]))
        else:
            dog_tr_txt.write("Dog_" + dog_train_img_buf[tr].split("/")[-1])
            dog_tr_txt.write(" ")
            dog_tr_txt.write("1")
            dog_tr_txt.write("\n")
            dog_tr_txt.flush()
            cv2.imwrite("E:/[1]DB/Cat_and_dog/Cat_Dog_500_train/" + "Dog_" + dog_train_img_buf[tr].split("/")[-1], dog_img)

    for v in range(len(cat_val_img_buf)):
        cat_img = cv2.imread(cat_val_img_buf[v], 1)
        dog_img = cv2.imread(dog_val_img_buf[v], 1)

        if cat_img is None:
            print("Cat image = {}".format(cat_val_img_buf[v].split("/")[-1]))
        else:
            cat_val_txt.write("Cat_" + cat_val_img_buf[v].split("/")[-1])
            cat_val_txt.write(" ")
            cat_val_txt.write("0")
            cat_val_txt.write("\n")
            cat_val_txt.flush()
            cv2.imwrite("E:/[1]DB/Cat_and_dog/Cat_Dog_100_val/" + "Cat_" + cat_val_img_buf[v].split("/")[-1], cat_img)

        if dog_img is None:
            print("Dog image = {}".format(dog_val_img_buf[v].split("/")[-1]))
        else:
            dog_val_txt.write("Dog_" + dog_val_img_buf[v].split("/")[-1])
            dog_val_txt.write(" ")
            dog_val_txt.write("1")
            dog_val_txt.write("\n")
            dog_val_txt.flush()
            cv2.imwrite("E:/[1]DB/Cat_and_dog/Cat_Dog_100_val/" + "Dog_" + dog_val_img_buf[v].split("/")[-1], dog_img)

    for t in range(len(cat_test_img_buf)):
        cat_img = cv2.imread(cat_test_img_buf[t], 1)
        dog_img = cv2.imread(dog_test_img_buf[t], 1)

        if cat_img is None:
            print("Cat image = {}".format(cat_test_img_buf[t].split("/")[-1]))
        else:
            cat_te_txt.write("Cat_" + cat_test_img_buf[t].split("/")[-1])
            cat_te_txt.write(" ")
            cat_te_txt.write("0")
            cat_te_txt.write("\n")
            cat_te_txt.flush()
            cv2.imwrite("E:/[1]DB/Cat_and_dog/Cat_Dog_200_test/" + "Cat_" + cat_test_img_buf[t].split("/")[-1], cat_img)

        if dog_img is None:
            print("Dog image = {}".format(dog_test_img_buf[t].split("/")[-1]))
        else:
            dog_te_txt.write("Dog_" + dog_test_img_buf[t].split("/")[-1])
            dog_te_txt.write(" ")
            dog_te_txt.write("1")
            dog_te_txt.write("\n")
            dog_te_txt.flush()
            cv2.imwrite("E:/[1]DB/Cat_and_dog/Cat_Dog_200_test/" + "Dog_" + dog_test_img_buf[t].split("/")[-1], dog_img)


if __name__ == "__main__":
    main()
