# DeepSkinning

This project runs with:
- Python 3.5.3
- Tensorflow 1.14.0
- Cuda Toolkit 10.0.130
- Blender 2.79.b (some other devices had some incompatibilites with Cuda & Tensorflow GPU)

How to train and run:

  Training: Select the models that you want our network models (LSTM, CNN or LSTM-CNN-Hybrid) in tensorflow_script.py (change network=....)
            You could change batch_size and epochs, as well. For training, you should always use start_tensorflow function in main.py.
            
  Use a pre-trained model: In order to use pre-trained network models you could use tensorflow_script_without_training script's called start_tensorflow and provide
                           the pre-trained model's name via main.py (.h5 files have been extracted from the above procedure).
                           
  P-Center: You are additionally able to use P-Center algorithm as the proxy bone estimation which is feasible by using in main.py the function p_center_clustering of
            p_center.py script. There you should also provide the number of bones (clusters). For comparison purposes of Deep Skinning with FESAM in both tensorflow_script
            & tensorflow_script_without_training we are calculating the number of bones that our network models decide.
 
  Plugin for Blender: Moreover, we have created a plugin for Blender for Show/Hide 3D models imported for training, 
                      link/unlink our method's output, change FPS of output animations & reload an empty scene for user's convenience.
![alt text](https://github.com/AnastasiaMoutafidou/DeepSkinning/blob/master/Plugin.PNG?raw=true)
