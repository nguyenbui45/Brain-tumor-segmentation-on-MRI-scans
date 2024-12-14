from PreprocessingData import Load_Brain_MRI_dataset, read_file_path


PATH = "/home/nguyensolbadguy/Code_Directory/DL_Implementation/computer_vision/Brain_MRI/"
train = 'train/'

train_img_list,train_mask_list = read_file_path(PATH,train)

instance = Load_Brain_MRI_dataset(train_img_list,train_mask_list,None,None)

