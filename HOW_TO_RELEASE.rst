How to release a new version
============================

1. Update version numbers in ``ramutils/__init__.py``
2. Update and rebuild documentation: ``python maint/build_docs.py``
3. Summarize changes in ``CHANGELOG.rst`` (you should be adding bullet points
   as you go anyway)
4. Merge into ``master`` after tests pass
5. Tag with the version number in the format ``v2.1.0``
6. Draft a new release on GitHub
7. Update the ``RAM_clinical`` account on ``rhino`` after consulting with
   Clinical Affairs
