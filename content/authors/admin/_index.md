---
# Display name
title: Jean Michel A. Sarr

# Name pronunciation (optional)


# Full name (for SEO)
first_name: Jean Michel Amath
last_name: Sarr

# Pronouns (optional)
pronouns: he/him

# Is this the primary user of the site?
superuser: true

# Role/position/tagline
role: Research Engineer

# Organizations/Affiliations to show in About widget
organizations:
  - name: Google
    url: https://www.google.com/

# Short bio (displayed in user profile at end of posts)
bio: A Research Engineer at Google focused on building and improving cutting-edge generative models, with a focus on synthetic data for model fine-tuning and robustness.

# Social Networking
# Need to use another icon? Simply download the SVG icon to your `assets/media/icons/` folder.
profiles:
  - icon: at-symbol
    url: 'mailto:jeanmichelamathsarr at gmail.com'
    label: E-mail Me
  - icon: brands/x
    url: https://x.com/jmamathsarr
    label: Follow on X
  - icon: brands/linkedin
    url: https://www.linkedin.com/in/jean-michel-amath-sarr/
  - icon: brands/google-scholar
    url: https://scholar.google.com/citations?user=qdrePlgAAAAJ&hl=en
  # Link to a PDF of your resume/CV - upload it to `static/uploads/resume.pdf`
  - icon: academicons/cv
    url: uploads/resume.pdf
    label: Download my resume

# Highlight the author in author lists? (true/false)
highlight_name: true

# Author's website URL
website: ""
---

I build infrastructure that accelerates research velocity at  Google.

As a Research Engineer, I specialize in designing systems that eliminate bottlenecks in LLM development. My work spans two complementary domains: infrastructure engineering and synthetic data research.

On the infrastructure side, I architect systems that decouple experimental logic from execution, achieving step-change improvements in research velocity. Currently, I'm building multimodal tool use infrastructure for Vision-Language Models (PaliGemma, Gemini, Gemma), enabling reliable visual instruction-following across computer vision tasks. Previously, I delivered 10x experiment acceleration (from ~10 to ~100 experiments per quarter) through consolidated, configuration-driven frameworks and scaled data loading capacity 15x by building robust TFDS infrastructure now serving 60+ datasets. I build systems where adding new experiments, datasets, or models incurs constant overhead rather than linear complexity.

On the research side, my expertise is synthetic data for post-training—specifically, how it scales to arbitrary domains when rigorous evaluation infrastructure enables tight iteration loops. As a core contributor to Gemini's multilingual capabilities, I architected the end-to-end synthetic data pipeline that scaled instruction-following across 25 languages. The quality came from making experiments cheap enough to run 50+ fine-tuning iterations, letting systematic hypothesis testing surface the right interventions rather than relying on manual curation. I cover generation for Supervised Fine-Tuning (SFT) and preference learning, with deep knowledge of synthetic alignment methods (including RLHF/RLAIF limitations and alternatives) synthesized in my research series.

This dual expertise is grounded in my PhD research at Sorbonne University, where I developed methods using synthetic data to predict model behavior under distribution shift—principles I now apply to designing robust, production-scale systems that unblock researchers at frontier labs.

I write about infrastructure design and the shift from human to synthetic labeling at jmamath.github.io.