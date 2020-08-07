# FedMAX
Source code for ECML-PKDD paper: FedMAX: Mitigating Activation Divergence for Accurate and Communication-Efficient Federated Learning

It implements experiments on three digit/object recognition datasets: FEMNIST*, CIFAR10, and CIFAR100 (both IID and non-IID)
It also implements experiments on two medical datasets: APTOS and ChestXray

## Run

Digit/object recognition datasets:
```commandline
bash digit_object_recognition/run.sh
```

Medical datasets:
```commandline
bash medical_data/aptos2019/run.sh
```

Synthetic data:
```commandline
bash digit_object_recognition/run_syn.sh
```


# Citation
@misc{chen2020fedmax,
    title={FedMAX: Mitigating Activation Divergence for Accurate and Communication-Efficient Federated Learning},
    author={Wei Chen and Kartikeya Bhardwaj and Radu Marculescu},
    year={2020},
    eprint={2004.03657},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
