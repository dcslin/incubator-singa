# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -ex

# anaconda login user name
USER=nusdbsystem
OS=$TRAVIS_OS_NAME-64

export PATH="$HOME/miniconda/bin:$PATH"
conda config --set anaconda_upload no

# save the package at given folder, then we can upload using singa-*.tar.bz2
suffix=$TRAVIS_JOB_NUMBER  #`TZ=Asia/Singapore date +%Y-%m-%d-%H-%M-%S`
export CONDA_BLD_PATH=~/conda-bld-$suffix
mkdir $CONDA_BLD_PATH

### DELETEME
TRAVIS_SECURE_ENV_VARS="true"
RELEASE_TYPE="patch"
### DELETEME

if [[ "$TRAVIS_SECURE_ENV_VARS" != "false" ]]
then
  git fetch -t
  VERSION=`git describe --abbrev=0 --tags`

  VERSION_BITS=(${VERSION//./ })

  VNUM1=${VERSION_BITS[0]}
  VNUM2=${VERSION_BITS[1]}
  VNUM3=${VERSION_BITS[2]}

  if [[ "$RELEASE_TYPE" == "major" ]]; then
    VNUM1=$((VNUM1+1))
  elif [[ "$RELEASE_TYPE" == "minor" ]]; then
    VNUM2=$((VNUM2+1))
  elif [[ "$RELEASE_TYPE" == "patch" ]]; then
    VNUM3=$((VNUM3+1))
  else
    echo "Release type is one of [major|minor|patch]"
    exit 1
  fi

  NEW_VERSION="$VNUM1.$VNUM2.$VNUM3"
  echo "Updating $VERSION to $NEW_VERSION"
  # github access
  # git tag $NEW_VERSION
  # git push --tags
  # echo "Tag created and pushed to github: $NEW_VERSION"
fi

conda build tool/conda/singa --python 3.6
conda build tool/conda/singa --python 3.7
# conda install --use-local singa
# cd test/python
# $HOME/miniconda/bin/python run.py

if [[ "$TRAVIS_SECURE_ENV_VARS" == "false" ]];
  # install and run unittest
then
  echo "no uploading if ANACONDA_UPLOAD_TOKEN not set"
else
  # turn off debug to hide the token in travis log
  set +x
  # upload the package onto anaconda cloud

  ANACONDA_UPLOAD_TOKEN="sh-c76ce9b7-676b-402b-ac05-32e140ae5221"
  USER="shicong"
  anaconda -t $ANACONDA_UPLOAD_TOKEN upload -u $USER -l main $CONDA_BLD_PATH/$OS/singa-*.tar.bz2 --force
fi
