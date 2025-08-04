---
layout: page
title: submitting
permalink: /submitting/
description:
nav: true
nav_order: 3
---

<!-- ### Template

The workflow you will use to participate in this track should be relatively familiar to you if have used [Github Pages](https://pages.github.com/). Specifically, our website uses the [Al-Folio](https://github.com/alshedivat/al-folio) template.
This template uses Github Pages as part of its process, but it also utilizes a separate build step using [Github Actions](https://github.com/features/actions) and intermediary [Docker Images](https://www.docker.com/).

**We recommend paying close attention to the steps presented in this guide. 
Small mistakes here can have very hard-to-debug consequences.**

### Contents

- [Quickstart](#quickstart)
- [Download the Blog Repository](#download-the-blog-repository)
- [Creating a Blog Post](#creating-a-blog-post)
- [Local Serving](#local-serving)
   - [Method 1: Using Docker](#method-1-using-docker)
   - [Method 2: Using Jekyll Manually](#method-2-using-jekyll-manually)
      - [Installation](#installation)
      - [Manual Serving](#manual-serving)
- [Submitting Your Blog Post](#submitting-your-blog-post)
- [Reviewing Process](#reviewing-process)
- [Camera Ready (TBD)](#camera-ready)


### Quickstart

This section provides a summary of the workflow for creating and submitting a blog post. 
For more details about any of these steps, please refer to the appropriate section.


1. Fork or download our [repository](https://github.com/data-brain-mind/blogpost-track#). 

2. Create your blog post content as detailed in the [Creating a Blog Post](#creating-a-blog-post) section.
    In summary, to create your post, you will: 
    - Create a Markdown or HTML file in the `_posts/` directory with the format `_posts/2025-04-28-[SUBMISSION NAME].md`. If you choose to write the post in HTML, then the extension of this last file should be .html instead of .md. NOTE: HTML posts are not officially supported, use at your own risk!
    - Add any static image to `assets/img/2025-04-28-[SUBMISSION NAME]/`.
    - Add any interactive HTML figures to `assets/html/2025-04-28-[SUBMISSION NAME]/`. 
    - Put your citations into a bibtex file in `assets/bibliography/2025-04-28-[SUBMISSION NAME].bib`. 

    **DO NOT** touch anything else in the repository.
    We will utilize an automated deployment action which will filter out all submissions that modifiy more than the list of files that we just described above.
    Read the [relevant section](#creating-a-blog-post) for more details.
    **Make sure to omit any identifying information for the review process.**

3. To render your website locally, you can build a docker container via `$ ./bin/docker_run.sh` to serve your website locally. 
    Alternatively, you can setup your local environment to render the website via conventional `$ bundle exec jekyll serve --future` commands. 
    More information for both of these configuratoins can be found in the [Local Serving](#local-serving) section.

4. To submit your website, create a pull request to the main repository. Make sure that this PR's title is `_posts/2025-04-28-[SUBMISSION NAME]`. This will trigger a GitHub Action that will build your blogpost and write the host's URL in a comment to your PR.

5. If accepted, we will merge the accepted posts to our main repository. See the [camera ready](#camera-ready) section for more details on merging in an accepted blog post.

**Should you edit ANY files other your new post inside the `_posts` directory, and your new folder inside the `assets` directory, your pull requests will automatically be rejected.**



### Download the Blog Repository

Download or fork our [repository](https://github.com/data-brain-mind/blogpost-track#). 
You will be submitting a pull request this repository. -->

### Creating a Blog Post in Markdown


To create your blog post in Markdown format, you can use [this example post](https://github.com/data-brain-mind/blogpost-track/blob/main/_posts/2025-04-28-distill-example.md) as a template. You can view the rendered version of the same post on our website [here]({% post_url 2025-04-28-distill-example %}).


<!-- You must modify the file's header (or 'front-matter') as needed.



 ```markdown
 ---
layout: distill
title: [Your Blog Title]
description: [Your blog post's abstract - no math/latex or hyperlinks!]
date: 2025-04-28
future: true
htmlwidgets: true

# anonymize when submitting 
authors:
  - name: Anonymous 

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton 

# must be the exact same name as your blogpost
bibliography: 2025-04-28-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: [Section 1]
  - name: [Section 2]
  # you can additionally add subentries like so
    subsections:
    - name: [Subsection 2.1]
  - name: [Section 3]
---

# ... your blog post's content ...
```

You must change the `title`, `discription`, `toc`, and eventually the `authors` fields (**ensure that the
submission is anonymous for the review process**). -->

<!-- Add any tags that are relevant to your post, such as the areas your work is relevant to. -->
Read our [sample blog post]({% post_url 2025-04-28-distill-example %}) carefully to see how you can add image assets, and how to write using $$\LaTeX$$!


**Important: make sure your post is completely anonymized before you export and submit it!**

### Submission Instructions
---

**Submission should be completed through the [OpenReview submission form](https://openreview.net/group?id=NeurIPS.cc/2025/Workshop/DBM/Tutorials&referrer=%5BHomepage%5D(%2F)#tab-your-consoles).**

To submit your tutorial, please follow these steps:

- **Markdown Content**: Enter your tutorial content into the **Markdown box** provided in the OpenReview submission form.
- **Figures**: Upload any images or figures used in your Markdown so they display correctly.
- **Code Links**: Include links to any relevant code (e.g., Jupyter notebooks or GitHub repositories). Be sure these links point to **anonymized** resources if required. These resources should demonstrate the core content of your tutorial.


You can preview how your tutorial will appear using the **Preview** tab in the OpenReview submission interface.


If you prefer to prepare your tutorial in a **Jupyter Notebook**, you may instead submit:

- A short overview of your tutorial in a `.md` (Markdown) file.
- A link to the corresponding `.ipynb` (notebook) file.

The Markdown file will be rendered and displayed on the blog page, along with a reference to your notebook.




---

Upon submission, we will review and render accepted blog posts on our  
[Blogpost Track blog page](https://data-brain-mind.github.io/blogpost-track/blog/index.html).

Our template is based on the [ICLR Blogposts 2025](https://iclr-blogposts.github.io/2025/about/) project.  
You can browse additional examples in their GitHub repository:  
[https://github.com/iclr-blogposts/2025/tree/main/_posts](https://github.com/iclr-blogposts/2025/tree/main/_posts)



### Camera-ready

**TBD** - instructions will be provided closer to the submission deadline.

### Full guide coming soon!