#
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
#!/bin/bash

if [ $# -eq 1 ]; then
  if [[ "$1" =~ ^(patch|minor|major)$ ]]; then
    echo "makeing release for $1"
  else
    echo "usage $ tool/release/release.sh [patch|minor|major]"
    exit 1
  fi
fi


#get highest tag number
VERSION=`git describe --abbrev=0 --tags`

#replace . with space so can split into an array
VERSION_BITS=(${VERSION//./ })

#get number parts and increase last one by 1
VNUM1=${VERSION_BITS[0]}
VNUM2=${VERSION_BITS[1]}
VNUM3=${VERSION_BITS[2]}

if [[ "$1" == "major" ]]; then
  VNUM1=$((VNUM1+1))
elif [[ "$1" == "minor" ]]; then
  VNUM2=$((VNUM2+1))
else
  VNUM3=$((VNUM3+1))
fi

#create new tag
NEW_TAG="$VNUM1.$VNUM2.$VNUM3"
echo "makeing release as tag $NEW_TAG"

#get current hash and see if it already has a tag
GIT_COMMIT=`git rev-parse HEAD`
CURRENT_COMMIT_TAG=`git describe --contains $GIT_COMMIT 2>/dev/null`

#only tag if no tag already (would be better if the git describe command above could have a silent option)
if [ -z "$CURRENT_COMMIT_TAG" ]; then
    echo "Updating $VERSION to $NEW_TAG"
    # git tag $NEW_TAG
    # git push --tags
    echo "..."
    echo "Tag created and pushed: $NEW_TAG"
else
    echo "This commit is already tagged as: $CURRENT_COMMIT_TAG"
fi

# build conda

# upload to cloud
