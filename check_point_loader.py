import elements

cp = elements.Checkpoint("/home/elksnis/logdir/20250716T225615/ckpt/20250717T095822F023824/agent.pkl")
cp.agent = agent
# We can also load checkpoints or parts of a checkpoint from a different directory.
cp.load("/home/elksnis/logdir/20250716T225615/ckpt/20250717T095822F023824/", keys=['agent'])
cp.load(keys=['agent'])
print(cp.model)
