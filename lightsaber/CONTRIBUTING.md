# Contributions

`Lightsaber` is developed and managed by a small team of Researchers and Engineers


## Contribution Guide

`Lightsaber` is being actively developed and will be updated rapidly. If you want to contribute to `Lightsaber` please follow the following rules

### Code Contribution Guide

We are following a git strategy as below:

- `master` always points to the latest stable release
- `dev` always points to current release candidate - typically the next release after the current stable version
- `master`, `dev`, and `gh-pages` branches are protected

  - `master` and `dev` will be maintained by admins
  - `gh-pages` is automatically maintained by `travis` - _avoid updating it manually_

- If you find bugs please create a github issue, with a small example and proper title

  - the maintainers would triage the issue, add the appropriate labels, and update it on the zenhub
  - If you have suggestions for change, please create a branch (if you have write access, else fork the repository) and create and pull request for the issue
  - maintainers would review the issue and approve the PR to dev

### Documentation Guide

Documentation is being rendered via `mkdocs` - we are following the principle of committing documentation as codes. 

* We are using `travis` to automatically build the documentation. 

  - commits to `dev` branch are rendered as the current `dev` documentation
  - commits to `master` branch are rendered as the current `stable` version of the documentation

* The documentation supports both `markdown` and `jupyter notebooks` which are rendered via `mkdocs`
* API is documented via the autodoc feature from `mkdocstrings`. To document a function/class, please follow the following steps:
  
  - Add documentation strings to your code. We are using `numpy` style documentation by default. [See](https://numpydoc.readthedocs.io/en/latest/format.html)
  - In the `docs/api.md` document, add the function/class you want to document. See existing example or the [external documentation from mkdocstrings](https://mkdocstrings.github.io/usage/#autodoc-syntax)
  - Any function class documented in the `docs/api.md` can be cross-referenced. [See how to do it here](https://mkdocstrings.github.io/usage/#cross-references)
* To add new sections, please follow the following syntax:

  - If you are adding a new file, place it under the appropriate level in the `mkdocs.yaml`
  - Within a `markdown`/`jupyter-notebook` file

    - top-level (`h1`) headings should be used only once at the begining of the document to provide a title for the chapter/section.
    - second-level (`h2`) headers are for sub-sections and will be rendered in the Table of contents for that chapter/section
    - Use sentence case for first to 3rd level headers (`h1, h2, h3`). Use capital case for 4th level onwards (`h4`)

## Release Update Guide

This section is mainly for the maintainers. 

- Once pull-requests to `dev` are reviewed and approved, accept the merge request to dev
  
  - Update the release candidate version number in `lightsaber.__version__.py` and in `.travis.yaml::DEV_VERSION`

- Based on milestones, upgrade `dev` to `master`

  - Update the stable  version number in `lightsaber.__version__.py` and in `.travis.yaml::STABLE_VERSION`

- It is ok for `dev` to be unstable and failing tests. `master` should always be in working state
