dir="step40000-unsharded"
addr="https://olmo-checkpoints.org/ai2-llm/olmo-medium/l6v218f4/step40000-unsharded"

mkdir -p ${dir}
cd ${dir}
wget ${addr}/config.yaml
wget ${addr}/model.pt
wget ${addr}/optim.pt
wget ${addr}/train.pt
cd ..


dir="step113000-unsharded"
addr="https://olmo-checkpoints.org/ai2-llm/olmo-medium/hrshlkzq/step113000-unsharded"

mkdir -p ${dir}
cd ${dir}
wget ${addr}/config.yaml
wget ${addr}/model.pt
wget ${addr}/optim.pt
wget ${addr}/train.pt
cd ..


dir="step339000-unsharded"
addr="https://olmo-checkpoints.org/ai2-llm/olmo-medium/cojbrc1o/step339000-unsharded"

mkdir -p ${dir}
cd ${dir}
wget ${addr}/config.yaml
wget ${addr}/model.pt
wget ${addr}/optim.pt
wget ${addr}/train.pt
cd ..