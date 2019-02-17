## CIFAR-100

### ResNet110
```
python main.py --model_name='resnet' --drop_rate=0.0 --drop_type='dropout';
python main.py --model_name='resnet' --drop_rate=0.1 --drop_type='dropout';
python main.py --model_name='resnet' --drop_rate=0.2 --drop_type='dropout';
python main.py --model_name='resnet' --drop_rate=0.3 --drop_type='dropout';
python main.py --model_name='resnet' --drop_rate=0.4 --drop_type='dropout';

python main.py --model_name='resnet' --drop_rate=0.1 --drop_type='Gaussian';
python main.py --model_name='resnet' --drop_rate=0.2 --drop_type='Gaussian';
python main.py --model_name='resnet' --drop_rate=0.3 --drop_type='Gaussian';
python main.py --model_name='resnet' --drop_rate=0.4 --drop_type='Gaussian';

python main.py --model_name='resnet' --drop_rate=0.1 --drop_type='Uniform';
python main.py --model_name='resnet' --drop_rate=0.2 --drop_type='Uniform';
python main.py --model_name='resnet' --drop_rate=0.3 --drop_type='Uniform';
python main.py --model_name='resnet' --drop_rate=0.4 --drop_type='Uniform';
```

### WideResNet28-10
```
python main.py --model_name='wideresnet' --drop_rate=0.0 --drop_type='dropout';
python main.py --model_name='wideresnet' --drop_rate=0.1 --drop_type='dropout';
python main.py --model_name='wideresnet' --drop_rate=0.2 --drop_type='dropout';
python main.py --model_name='wideresnet' --drop_rate=0.3 --drop_type='dropout';
python main.py --model_name='wideresnet' --drop_rate=0.4 --drop_type='dropout';

python main.py --model_name='wideresnet' --drop_rate=0.1 --drop_type='Gaussian';
python main.py --model_name='wideresnet' --drop_rate=0.2 --drop_type='Gaussian';
python main.py --model_name='wideresnet' --drop_rate=0.3 --drop_type='Gaussian';
python main.py --model_name='wideresnet' --drop_rate=0.4 --drop_type='Gaussian';
```
