#!/bin/bash
source /home/elksnis/anaconda3/etc/profile.d/conda.sh
conda activate dreamer3

set -e

if [ ! -d "lab" ]; then
  git clone https://github.com/google-deepmind/lab.git
fi
#git clone https://github.com/google-deepmind/lab.git
cd lab

cat > .bazelrc << EOL
build --enable_workspace
build --python_version=PY3
build --action_env=PYTHON_BIN_PATH=$(which python3)
EOL

sed -i '/\[py_binary(/,/\]]/c\
py_binary(\
    name = "python_game_py3",\
    srcs = ["examples/game_main.py"],\
    data = ["//:deepmind_lab.so"],\
    main = "examples/game_main.py",\
    python_version = "PY3",\
    srcs_version = "PY3",\
    tags = ["manual"],\
    deps = ["@six_archive//:six"],\
)' BUILD

cat > python_system.bzl << 'EOL'
_BUILD_FILE = '''

exports_files(["defs.bzl"])

cc_library(
    name = "python_headers",
    hdrs = glob(["python3/**/*.h", "numpy3/**/*.h"]),
    includes = ["python3", "numpy3"],
    visibility = ["//visibility:public"],
)
'''

_GET_PYTHON_INCLUDE_DIR = """
import sys
from distutils.sysconfig import get_python_inc
from numpy import get_include
sys.stdout.write("{}\\n{}\\n".format(get_python_inc(), get_include()))
""".strip()

def _python_repo_impl(repository_ctx):
    repository_ctx.file("BUILD", _BUILD_FILE)
    result = repository_ctx.execute(["python3", "-c", _GET_PYTHON_INCLUDE_DIR])
    if result.return_code:
        fail("Failed to run local Python3 interpreter: %s" % result.stderr)
    pypath, nppath = result.stdout.splitlines()
    repository_ctx.symlink(pypath, "python3")
    repository_ctx.symlink(nppath, "numpy3")

python_repo = repository_rule(
    implementation = _python_repo_impl,
    configure = True,
    local = True,
    attrs = {"py_version": attr.string(default = "PY3", values = ["PY3"])},
)
EOL

bazel clean --expunge
bazel build -c opt //python/pip_package:build_pip_package --verbose_failures

./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
pip install --force-reinstall /tmp/dmlab_pkg/deepmind_lab-*.whl
DEST="$(python3 -c 'import site; print(site.getsitepackages()[0])')/deepmind_lab"
mv "${DEST}/_main/"* "${DEST}/"
rmdir "${DEST}/_main"
cd ..
#rm -rf lab

if [ -d "dmlab_data" ]; then
  rm -rf dmlab_data
fi

mkdir dmlab_data
cd dmlab_data
pip install Pillow
curl https://bradylab.ucsd.edu/stimuli/ObjectsAll.zip -o ObjectsAll.zip
unzip ObjectsAll.zip
cd OBJECTSALL
python3 << EOM
import os
from PIL import Image
files = [f for f in os.listdir('.') if f.lower().endswith('jpg')]
for i, file in enumerate(sorted(files)):
  print(file)
  im = Image.open(file)
  im.save('../%04d.png' % (i+1))
EOM
cd ..
rm -rf __MACOSX OBJECTSALL ObjectsAll.zip
DMLAB_DATA=$(pwd | sed 's/[\/&]/\\&/g')
sed -i "s|DATASET_PATH = ''|DATASET_PATH = '$DMLAB_DATA'|g" "$DEST/baselab/game_scripts/datasets/brady_konkle_oliva2008.lua"
cd ..

python3 -c "import deepmind_lab; deepmind_lab.Lab('contributed/dmlab30/explore_goal_locations_small', []).reset();"
python3 -c "import deepmind_lab; deepmind_lab.Lab('contributed/dmlab30/psychlab_arbitrary_visuomotor_mapping', []).reset();"

echo "DMLab installed"
