How to release a new version
============================

1. Update version numbers in ``ramutils/__init__.py``
2. Update and rebuild documentation: ``python maint/build_docs.py``
3. Summarize changes in ``CHANGELOG.rst`` (you should be adding bullet points
   as you go anyway)
4. Run the full set of tests as outlined in the README
5. Confirm that config files produced during testing can be loaded
   into RAMulator without issue. This involves copying zipped config files
   onto a flash drive and loading them one at a time from a host pc
6. Merge the pre-release branch into ``master`` after tests pass
7. Create a new release on GitHub with a version number in the format ``v2.1.0``
8. Build and upload the conda package ``python maint/build.py --upload``
9. Notify users of the new release and include a link to the release notes on Github
10. Update the ``RAM_clinical`` account on ``rhino`` to use the new release
