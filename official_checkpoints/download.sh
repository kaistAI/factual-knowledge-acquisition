# dir="1b/step117850-unsharded"
# addr="https://olmo-checkpoints.org/ai2-llm/olmo-small/czmq3tph/step117850-unsharded"

dir="7b/step50000-unsharded"
addr="https://olmo-checkpoints.org/ai2-llm/olmo-medium/l6v218f4/step50000-unsharded"


mkdir -p ${dir}
cd ${dir}
wget ${addr}/config.yaml
wget ${addr}/model.pt
wget ${addr}/optim.pt
wget ${addr}/train.pt
cd ..