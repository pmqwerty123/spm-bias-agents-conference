# Third-party dependencies and data

The simulation layer uses NumPy, Pydantic, the OpenAI Python client, and pytest.
They are installed separately, not vendored here. Their respective licenses and
terms remain applicable; this file does not replace those licenses.

The historical experiments used the MOSAIC persona bank and a compatible database
schema. MOSAIC source code and persona records are not included in this release.
The inspected local upstream copy did not include a top-level LICENSE file; this
observation is not a determination of the current upstream project's full terms.
Verify current permissions and data-specific terms before downloading or sharing.

MOSAIC project: https://github.com/genglinliu/MOSAIC

Associated publication: Liu et al., *MOSAIC: Modeling Social AI for Content
Dissemination and Regulation in Multi-Agent Simulations*, EMNLP 2025.
https://aclanthology.org/2025.emnlp-main.325/

The built-in fallback persona sketches in `src/spm/personas.py` are documented
as synthetic testing fixtures, not the survey-derived bank used in paper runs.