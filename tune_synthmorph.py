import os
import sys
import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
import voxelmorph as vxm
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from data_generator import SynthradData
from tensorflow.keras.callbacks import ModelCheckpoint


# Function to convert PyTorch tensors to TensorFlow tensors
def convert_to_tf_dataset(loader):
    for ct_fixed, mr_fixed, ct_moving, mr_moving, transform, inv_transform in loader:
        yield ((tf.convert_to_tensor (ct_moving.squeeze(-1).numpy()),
                tf.convert_to_tensor (mr_fixed.squeeze(-1).numpy()),),
               tf.convert_to_tensor (transform.squeeze(-1).numpy()),
               )

        #yield (tf.convert_to_tensor (ct_fixed.numpy()),
        #       tf.convert_to_tensor (mr_fixed.numpy()),
        #       tf.convert_to_tensor (ct_moving.numpy()),
        #       tf.convert_to_tensor (mr_moving.numpy()),
        #       tf.convert_to_tensor (transform.numpy()),
        #       tf.convert_to_tensor (inv_transform.numpy()))

def inverse_loss (target, predicted):
    predicted = tf.concat ((predicted, tf.constant([[[0,0,0,1]]], dtype=tf.float32)), axis=1)
    return tf.norm (tf.linalg.matmul (target, predicted))

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    epochs = 20
    num_workers = 0

    data_folder = sys.argv[1]
    val_folder = sys.argv[2]
    rigid = sys.argv[3].lower() in ("rigid", "true", "1", "yes")

    save_folder = data_folder + "finetune_rigid/" if rigid else data_folder + "finetune_affine/"
    os.makedirs (save_folder, exist_ok=True)

    train_data = SynthradData (data_folder,
                               include_mask=False)
    val_data = SynthradData (val_folder,
                             include_mask=False)

    dataloader = DataLoader(train_data,
                            batch_size=1,
                            shuffle=True,
                            num_workers=num_workers)
    valloader = DataLoader(val_data,
                           batch_size=1,
                           shuffle=False,
                           num_workers=num_workers)

    in_shape = train_data.shape ()
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: convert_to_tf_dataset(dataloader),
        output_signature=((tf.TensorSpec(shape=(None,)+in_shape, dtype=tf.float32),
                           tf.TensorSpec(shape=(None,)+in_shape, dtype=tf.float32),),
                          tf.TensorSpec(shape=(None, 4, 4), dtype=tf.float32)))
    tf_dataset = tf_dataset.repeat()

    val_dataset = tf.data.Dataset.from_generator(
        lambda: convert_to_tf_dataset(valloader),
        output_signature=((tf.TensorSpec(shape=(None,)+in_shape, dtype=tf.float32),
                           tf.TensorSpec(shape=(None,)+in_shape, dtype=tf.float32),),
                          tf.TensorSpec(shape=(None, 4, 4), dtype=tf.float32)))
    #val_dataset = val_dataset.repeat()


    model = vxm.tf.networks.VxmAffineFeatureDetector(in_shape, rigid=rigid, make_dense=False)

    if rigid:
        #model_aff.load_weights ("synthmorph.rigid.1.h5")
        model.load_weights ("./freesurfer/models/synthmorph_rigid.h5")
    else:
        model.load_weights ("./freesurfer/models/synthmorph_affine.h5")
        #model.load_weights ("synthmorph.affine.crop.h5")

    model.compile (optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=inverse_loss)

    model_save = save_folder + "synthmorph_epoch{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath=model_save,
                                 save_weights_only=False,
                                 monitor='val_loss',
                                 mode='min',
                                 save_best_only=False
                                 )

    model.fit (tf_dataset,
               epochs=epochs,
               steps_per_epoch=len(dataloader),
               validation_data=val_dataset,
               callbacks=[checkpoint])

    #model.fit (tf_dataset,
    #           epochs=epochs,
    #           steps_per_epoch=5,
    #           validation_data=val_dataset,
    #           callbacks=[checkpoint])

    #model.save(save_folder + "synthmorph_model.h5")

    np.save(save_folder + "loss_history.npy", np.array(model.history.history["loss"]))
    np.save(save_folder + "val_history.npy", np.array(model.history.history["val_loss"]))


    plt.figure()
    plt.plot(range(1, epochs + 1), model.history.history["loss"], linestyle="-", label="Training Loss")
    plt.plot(range(1, epochs + 1), model.history.history["val_loss"], linestyle="-", label="Validation Loss")

    plt.title("Training and Validation Loss during training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.savefig(save_folder + "training_loss.png")
