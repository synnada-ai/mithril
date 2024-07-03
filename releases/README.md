# Releases

This folder contains information about all currently released Mithril versions. You can check what has changed in each release in this folder.


## Steps

1. Create a branch with name of "prepare_{x.y.z}"
2. Checkout the branch
3. Update setup.py with the correct version, check also other informations
4. Push changes in setup.py
5. Change directory to releases directory
6. Ensure all items are done in the checklist below
7. Run the command ```bash push_release_log.sh <version> <token>```


## Release Checklist

The person responsible for the upcoming release of Mithril should follow this checklist:

- [ ] Check that all examples run without issues.
- [ ] Ensure that all README models run without issues.
- [ ] Re-check all newly added features and ensure they are well-tested.
- [ ] Verify that the version number is correct.
- [ ] Run speed benchmarks. Ensure that pure framework models are not significantly faster than Mithril models.
- [ ] Run `cProfile` in the library and, if possible, compare it with the previous release.
- [ ] Verify all distributions in the release, ensuring that all tests pass.