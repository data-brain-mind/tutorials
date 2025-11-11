Instructions of GitHub Workflow (Option 2\)

Step 1\. Reference Template

* **Live example:** [https://data-brain-mind.github.io/tutorials/blog/2025/distill-example/](https://data-brain-mind.github.io/tutorials/blog/2025/distill-example/)  
* **Source file:** \_posts/2025-04-28-distill-example.md  
* **Bibliography:** assets/bibliography/2025-04-28-distill-example.bib

This example illustrates:

* Proper frontmatter (title, authors, affiliations)  
* Section structure and table of contents  
* Figure inclusion with {% include figure.html ... %}  
* Citation format with \<d-cite key="..."\>\</d-cite\>  
* Code block and equation formatting

Step 2\. Locate Your Tutorial Files

You may modify **only** the following files:

* \_posts/2025-11-24-accelerated-methods-in-multi-modal-multi-metric-many-model-cogneuroai.md  
* assets/img/2025-11-24-accelerated-methods-in-multi-modal-multi-metric-many-model-cogneuroai/  
* assets/bibliography/2025-11-24-accelerated-methods-in-multi-modal-multi-metric-many-model-cogneuroai.bib

Step 3\. Test Locally (optional)

./bin/docker\_run.sh

\# Or:

bundle exec jekyll serve \--future

Then open [http://localhost:4000/tutorials/](http://localhost:4000/tutorials/) to preview your post.

For more information, refer to [this instruction](https://github.com/data-brain-mind/tutorials).

Step 4\. Create and Submit a Pull Request

git checkout \-b camera-ready-2025-11-24-accelerated-methods-in-multi-modal-multi-metric-many-model-cogneuroai

git add .

git commit \-m "Camera-ready version for Accelerated Methods in {Multi-Modal, Multi-Metric, Many-Model} CogNeuroAI"

git push origin camera-ready-2025-11-24-accelerated-methods-in-multi-modal-multi-metric-many-model-cogneuroai

Then open a pull request:

* Go to [https://github.com/YOUR-USERNAME/tutorials](https://github.com/YOUR-USERNAME/tutorials)  
* Click **“Compare & pull request”**  
* Set base repository to data-brain-mind/tutorials and base branch to main  
* Title: *Camera-ready: Accelerated Methods in {Multi-Modal, Multi-Metric, Many-Model} CogNeuroAI*  
* Add a brief description of your updates  
* Click **“Create pull request”**

For more details, please see GitHub’s documentation:

[Creating a pull request from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)

We look forward to receiving your camera-ready submission.

