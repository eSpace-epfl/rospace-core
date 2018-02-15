## Contributing to Source Code

Thanks for wanting to contribute source code to the RDV Simulator. That's great!

### Documentation

All code should be documented using the [Google Python Style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and should be
processable by `pdoc` (A documentation generator meant to replace epydoc).

### Tests

In order to constantly increase the quality of our software we can no longer accept pull request which submit un-tested code.
It is a must have that changed and added code segments are unit tested.
In some areas unit testing is hard (aka almost impossible) as of today - in these areas refactoring WHILE fixing a bug is encouraged to enable unit testing.

### Sign your work

We use the Developer Certificate of Origin (DCO) as a additional safeguard
for the RDV Simulator project. This is a well established and widely used
mechanism to assure contributors have confirmed their right to license
their contribution under the project's license.
Please read [docs/contribute/developer-certificate-of-origin][dcofile].
If you can certify it, then just add a line to every git commit message:

````
  Signed-off-by: Random J Developer <random@developer.example.org>
````

Use your real name (sorry, no pseudonyms or anonymous contributions).
If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`. You can also use git [aliases](https://git-scm.com/book/tr/v2/Git-Basics-Git-Aliases)
like `git config --global alias.ci 'commit -s'`. Now you can commit with
`git ci` and the commit will be signed.

### Apply a license

In case you are not sure how to add or update the license header correctly please have a look at [docs/contribute/how-to-apply-a-license.md][applyalicense]
