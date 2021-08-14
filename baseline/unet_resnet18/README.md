# Baseline2
Despite being a folder for the unet_resnet18 model, this folder also houses code for a EfficientnetB8-FPN Model. You can specify model architecture during training.

To train this model you will need to run the set up scripts found within the data_utils folder. After which you can run a training instance with the following command

```zsh
python main.py --data-directory ../../data/EndGame --exp_directory EndGame_Unet_Model --batch-size 1 --epochs 30 --arch Unet 
```

If you are interested in running inference with the model, you can use the provided bash script run_model.sh. Open the file to see the nature of the arguments. An example call of this script is as follows:

```zsh
./run_model.sh ../../samples/image/target FPN_Out EndGame_FPN_Model/best_weights.pt 0.60 Efficientnetb8_FPN
```
