import os

NUM_CFGS = 12
INSTANCE_NAME_BASE = 'nips-exp-'
BUCKET = 'your-bucket'
CODE_TAR = 'exp_1024/code.tar.gz'
CFG_BASE = 'exp_1024/exp_yaml/'

with open('gcloud_cmd.template', 'r') as f:
    template_cmd = f.readline().replace('\n', '')

# Fill in stuff common to all cfgs
template_cmd = template_cmd.replace('bucket-ph', BUCKET)
template_cmd = template_cmd.replace('code-ph', CODE_TAR)

cmds = []

for i in range(NUM_CFGS):
    cmd = template_cmd.replace('instance-name-ph', INSTANCE_NAME_BASE + 'cfg%d' % i)
    cmd = cmd.replace('cfg-ph', CFG_BASE + 'cfg%d.yaml' % i)
    cmds.append(cmd)

with open('gcloud_cmd', 'w') as f:
    for cmd in cmds:
        f.write(cmd + '\n')
