# ssh
- client(文書機,your computer) v.s. server

1. log in to server by password (on server)
1. making key and lock (on server)
    >```
    >cd ~/.ssh
    >ssh-keygen -t ed25519 -f ~/.ssh/ed25519_zinfandel_m5813 -C "zinfandel_m5813_test"
    >chmod 600 ~/.ssh/ed25519_zinfandel_m5813_test
    >cat ~/.ssh/ed25519_zinfandel_m5813_test.pub >> ~/.ssh/authorized_keys
    >chmod 600 ~/.ssh/authorized_keys
    ># ~/.ssh/id_rsa是私鑰 用usb傳到client(文書機)
    ># ~/.ssh/id_rsa.pub是公鑰
    >```
    scp ~/.ssh/ed25519_zinfandel_m5813_test m5-cluster:~/.ssh/

1. check /etc/ssh/sshd_config (on server)
    >```
    >nano /etc/ssh/sshd_config
    >```

    in /etc/ssh/sshd_config
    >``` 
    >PubkeyAuthentication yes
    >PasswordAuthentication no
    >```

    restart
    >```
    >sudo systemctl restart sshd
    >```
1. using usb to put ~/.ssh/id_rsa to client ~/.ssh (server -> client)
1. making ~/.ssh/config (on client)
    >```
    >nano ~/.ssh/config
    >en
    >Host <server_name>
    >  HostName <server_ip>
    >  Port <port>
    >  User <user_name>
    >  IdentityFile ~/.ssh/id_rsa
    >```

# git
1. (on server) download mdgen-tutorial-code by git 
    >```
    >git clone https://github.com/32176701Jeff/mdgen-teaching.git
    >```

1. 常常需要備份 但是比如output或是ckpt這個資料夾裡面的data比較大 太大的話git無法備份 所以要改一下.gitignore裡面增加output ckpt這兩個資料夾
    >```
    >cd mdgen-teaching
    >nano .gitignore
    >output
    >ckpt
    >```
1. git add and git commit
    >```
    >git add .
    >git commit -m "2025-11-01 11:21"
    >```

# conda environment of mdgen
1. environment install
    >```
    >conda create -n mdgen python=3.8
    >conda activate mdgen
    >
    >conda install pandas -c conda-forge
    >conda install numpy -c pypi
    >conda install scipy -c pypi
    >
    >conda install matplotlib -c pypi
    >conda install seaborn -c conda-forge
    >conda install scikit-image -c pypi
    >conda install scikit-learn -c conda-forge
    >
    >conda install mdtraj -c conda-forge
    >
    >conda install pytorch -c pypi
    >conda install torchmetrics -c pypi
    >conda install pytorch-lightning -c pypi
    >
    >conda install aiohttp -c pypi
    >conda install requests -c pypi
    >
    >conda install jupyter -c conda-forge
    >conda install ipython -c conda-forge
    >conda install jupyter-client -c conda-forge
    >
    >conda install absl-py -c pypi
    >conda install matplotlib-base -c conda-forge
    >conda install tqdm -c conda-forge
    >conda install pip
    >conda install setuptools
    >conda install pydantic -c pypi
    >
    >```