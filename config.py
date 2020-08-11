hp = {
    'net_type': "DCGAN",
    'total_train_images': 200000,
    'train_data_path': " ",  # set your data set path here eg: "../datasets/CelebA/img_align_celeba"
    'output_dir': "./",
    # train settings
    "num_epochs": 50,
    "learning_rate": 2e-4,
    "batch_size": 64,
    "num_workers": 16,
    "beta1": 0.5,
    "beta2": 0.999,
    'save_image_dir': "GeneratedImages"
}