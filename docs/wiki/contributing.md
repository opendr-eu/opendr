# Contributing to OpenDR Toolkit

OpenDR Toolkit welcomes contributions to our open source project on GitHub.
When contributing please follow the OpenDR Code of Conduct.

## Issues

Feel free to submit issues and enhancements requests.

Please check if an bug report is already available to avoid duplicating issues.

## Contributing

Please refer to the coding style specific to your contributions.

### OpenDR partners' contributions

Every commit to the `master` branch should be done through a moderated Pull Request.

#### Regular Workflow

``` shell
# setup the git repository
cd ~/develop
git clone https://github.com/opendr-eu.git
cd opendr-eu

while (working) {
  # Bug fixes, optimizations, clean-up
  # New features, changes breaking binary compatibility, code refactoring, etc.
  # Enter in the master branch
  git checkout master

  # Update the master branch
  git fetch origin
  git merge origin/master [master]

  # Create a new feature branch
  git checkout -b <my-new-feature>

  # verify the branches list
  if (required)
    git branch

  # modify and commit files
  while (required) {

    while (required) {
      -> modify files
      git diff [<modified-files>]
      git status
    }

    # added files to the stage
    git add [<modified-files>]

    # monitor the already committed files
    if (required)
      git diff --staged [<modified-files>]

    # commit files
    # note: commits are local
    git commit [<modified-files>] -m "<My detailed comment>"

    # push the branch on the remote repository if required
    if (someTimes or lastCommit) {
      git push origin <my-new-feature>

      -> Look through the modifications in the GitHub GUI
    }

    # Create the Pull Request (PR)
    # this is can be done at an early development stage in order to open the
    # discussion with the colleagues, share a TODO list and/or a description
    # of the aims. In such a case, the PR should be opened as a draft PR.
    if (someTimes or lastCommit) {
      -> From the GitHub GUI: find the "Compare and Pull Request" button
      put origin/<my-new-feature> into origin/master
    }

    if (lastCommit) {
      -> If the github "Merge" button is unavailable because GitHub detected
         a merge conflict, the developer should fix it:
      if (mergeButtonDisabled)
        continue
      -> The developer should check by how many commits her branch is behind the
         develop (displayed as "behind" on the GitHub branches page) and evaluate
         the risk of bugs arising from possible logic conflicts (as opposed to merge
         conflicts which are well managed by GitHub ).
      if (developerUnsureAboutPossibleLogicConflict) {
        -> The developer should merge the current develop with her branch:
        git merge origin/master
        git push origin <my-new-feature>
        -> and test her code again, if the test fails, she should fix the problems.
        if (testFail)
          continue
      }

      -> The developer can mark the PR as ready for review (if the PR was a draft)
         and ask for a review (to a opendr-eu/development-team member, or a specific
         developer, or several specific developers) when the PR is completed and ready
         for review.
      -> The reviewer can post a review icon (":eyes:") when he starts the review
         (to avoid that multiple developers review at the same time) and start the
         review.
      -> When a PR is approved, the author can merge it (if nobody else marked that he
         is reviewing the PR)
      -> If more than one review is required, a comment should be posted asking for a
         second (or third reviewer) once the first reviewer accepted the PR.
      if (the PR can be merged) {
        # The developer has the responsibility to accept/delete the remote/local branches
        # The developer accepts the PR from the GitHub GUI

        if (wantToDelete) {
          # delete the remote branch (GitHub has also a button in the PR to do that)
          git push origin :<my-new-feature>

          # Delete the local branch
          git checkout master
          git branch -d <my-new-feature> # use '-D' instead of '-d' to delete an
                                         # un-merged branch
          break
        }
      }
      else { // not accepted
        The reviewer should explain what's wrong with the PR and use as much as
        possible 'change suggestions'.
      }
    }
  }
}
```

#### Setting up your identity

You need to setup your identity so that your pushes are correctly identified and appear under your name on github:

``` bash
git config --global user.email "me@here.com"
git config --global user.name "Billy Everyteen"
```
additionally you can cache your username and password so that you don't have to type them every time you contact github:
``` bash
git config --global credential.helper cache
```

More information about set up git can be found at https://help.github.com/articles/set-up-git.

### External contributions

In general, we follow the "fork-and-pull" Git workflow.
1. **Fork** the repo on GitHub
2. **Clone** the project to your own machine
3. **Commit** changes to your own branch
4. **Push** your work back up to your fork
5. Submit a **Pull request** so that we can review your changes

NOTE: Be sure to merge the latest from "upstream" before making a pull request!


## Copyright and Licensing

OpenDR Toolkit is licensed under the Apache 2.0 license.
OpenDR project does require that you make contributions available under the Apache 2.0 license in order to be included in the main repo.
If appropriate, include the Apache 2.0 license summary at the top of each file along with the copyright info.
If you are adding a new file that you wrote, include your name in the copyright notice in the license summary at the top of the file.

OpenDR project doesn't require to assign the copyright of your contributions, you retain the copyright.

### License Summary

You can copy and paste the Apache 2.0 license summary from below

```
// Copyright 2020-2021 OpenDR European Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
```
