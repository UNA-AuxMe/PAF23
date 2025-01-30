# Documentation Requirements

**Summary:** This document outlines the guidelines for documentation to ensure consistency, readability, maintainability, and adherence to best practices in coding standards.

- [1. In Code Documentation](#1-in-code-documentation)
  - [1.1. Linting](#11-linting)
    - [1.1.1. Python Linting](#111-python-linting)
    - [1.1.2. Markdown Linting](#112-markdown-linting)
  - [1.2. Templates](#12-templates)
  - [1.3. Python Docstrings](#13-python-docstrings)
  - [1.4. Readability and Maintainability](#14-readability-and-maintainability)
  - [1.5. Code Structure](#15-code-structure)
  - [1.6. Efficiency and Performance](#16-efficiency-and-performance)
  - [1.7. Error Handling](#17-error-handling)
  - [1.8. Testing](#18-testing)
  - [1.9. Security](#19-security)
  - [1.10. Version Control](#110-version-control)
  - [1.11. Scalability](#111-scalability)
  - [1.12. Consistency with Coding Standards](#112-consistency-with-coding-standards)
- [2. Markdown Documentation](#2-markdown-documentation)
  - [2.1. Structure and Organization](#21-structure-and-organization)
  - [2.2. Content Detailing](#22-content-detailing)
  - [2.3. Usage Documentation](#23-usage-documentation)
  - [2.4. Visual Aids](#24-visual-aids)
  - [2.5. Consistency and Style](#25-consistency-and-style)
  - [2.6. Maintenance](#26-maintenance)

## 1. In Code Documentation

In-code documentation is essential for maintaining code quality, readability, and maintainability. It helps developers understand the purpose of the code, its functionality, and how to use it effectively. Below are the requirements for in-code documentation:

### 1.1. Linting

You can lint your files locally with:

```bash
docker compose -f build/docker-compose.linting.yaml up
```

#### 1.1.1. Python Linting

Ensure that all Python files pass linting checks without issues.
Tools such as `black` are already present in this project.
Regular linting ensures that the code adheres to the project's style guide, enhancing readability and maintainability.

#### 1.1.2. Markdown Linting

Lint all markdown files with `markdownlint`.

### 1.2. Templates

Follow the guidelines presented in the following templates:

- [python](templates/template_class.py)

### 1.3. Python Docstrings

The project comes with an extension to automatically generate docstrings. It is mandatory to use it if you comment your code.

To generate a docstring you have to be in the development container and then a popup will apear after writing three quotation marks:

![docstring_popup.png](/doc/assets/development/docstring_popup.png)

Press `Enter` or select the option and it should produce a docstring that looks like this:

![docstring.png](/doc/assets/development/docstring.png)

### 1.4. Readability and Maintainability

- **Consistent Formatting:** Code should follow a consistent and readable formatting style. Tools like linters or formatters can help enforce a consistent code style.
  - [linting](./linting.md)
- **Meaningful Names:** Variable and function names should be descriptive and convey the purpose of the code.
- **Comments:** Clear and concise comments should be used where necessary to explain complex logic or provide context.

### 1.5. Code Structure

- **Modularity:** Code should be organized into modular components or functions, promoting reusability and maintainability.
- **Appropriate Use of Functions/Methods:** Functions should have a clear purpose and adhere to the single responsibility principle.
- **Hierarchy and Nesting:** Avoid overly nested structures; use appropriate levels of indentation to enhance readability.

### 1.6. Efficiency and Performance

- **Optimized Algorithms:** Code should use efficient algorithms and data structures to achieve good performance.
- **Avoidance of Code Smells:** Detect and eliminate code smells such as duplicated code, unnecessary complexity, or anti-patterns.

### 1.7. Error Handling

- **Effective Error Messages:** Error messages should be clear and provide useful information for debugging.
- **Graceful Error Handling:** Code should handle errors gracefully, avoiding crashes and providing appropriate feedback.

### 1.8. Testing

- **Comprehensive Test Coverage:** Code should be accompanied by a suite of tests that cover different scenarios, ensuring reliability and maintainability.
- **Test Readability:** Tests should be clear and easy to understand, serving as documentation for the codebase.

### 1.9. Security

- **Input Validation:** Code should validate and sanitize inputs.

### 1.10. Version Control

- **Commit Messages:** Use descriptive and meaningful commit messages to track changes effectively.
- **Branching Strategy:** Follow a consistent and well-defined branching strategy to manage code changes.

### 1.11. Scalability

- **Avoid Hardcoding:** Parameterize values that might change, making it easier to scale the application.
  - In ROS noetic, you can use the [`rosparam`](https://wiki.ros.org/rosparam) command to set parameters and [`rospy Parameter Server`](https://wiki.ros.org/rospy/Overview/Parameter%20Server) to retrieve them.
  - Additionally, you can use the [`roslaunch`](https://wiki.ros.org/roslaunch) command to launch nodes with parameters.
  - Furthermore, you can use [`dynamic_reconfigure`](https://wiki.ros.org/dynamic_reconfigure) to change parameters at runtime and [`rqt_reconfigure`](https://wiki.ros.org/rqt_reconfigure) for a graphical interface.
- **Modular Design:** Use a modular design to facilitate the addition of new features and components.
- **Optimized Resource Usage:** Ensure efficient utilization of resources to support scalability.

### 1.12. Consistency with Coding Standards

- **Adherence to Coding Guidelines:** Follow established coding standards and best practices for the programming language or framework used.

## 2. Markdown Documentation

- **Code Comments:** In addition to in-code comments, consider external documentation for the overall project structure, APIs, usage instructions and configurations.
- **README Files:** Include a well-written README file that provides an overview of the project, installation instructions, and usage examples.

To effectively document complex technical aspects of your project, it's essential to provide clear, structured, and comprehensive Markdown documentation.
This ensures that both current and future team members, as well as external viewers, can understand and utilize the project's components effectively.
Below are the requirements for creating such documentation:

### 2.1. Structure and Organization

- **Hierarchical Headings:** Utilize headings to create a clear hierarchy, starting with a single `#` for main titles and increasing the number of `#` symbols for subheadings. This structure enhances readability and allows readers to navigate the document easily.
- **Table of Contents:** For all documents, include a table of contents at the beginning, linking to major sections. This facilitates quick access to different parts of the documentation.
- **Templates**: Use the [markdown](templates/template_wiki_page.md) template.
- **Number Headings**: Use the command `Ctrl + Shift + P` and search for `Markdown All in One: Add/Update section numbers` to number the headings.

### 2.2. Content Detailing

- **Comprehensive Explanations:** Provide detailed explanations of complex technical concepts, ensuring that all relevant aspects are covered thoroughly. Avoid assuming prior knowledge beyond the basics, and define specialized terms when first introduced.
- **Link prior knowledge**: If you are referencing a concept that was previously explained, or assume any general knowledge, link to the explanations.
- **Step-by-Step Instructions:** When outlining procedures or workflows, present information in a sequential, step-by-step format. Numbered lists can guide the reader through processes systematically.

### 2.3. Usage Documentation

- **Explicit Usage Instructions:** Clearly describe how to use each component, feature, or module. Include practical examples and scenarios to illustrate typical use cases, ensuring users can apply the information effectively.
- **Code Examples:** Incorporate code snippets where applicable, enclosed in triple backticks (```) with the appropriate language identifier for syntax highlighting. This aids in demonstrating usage in a practical context.

An example that also includes a command to lint the code and markdown files:

```bash
docker compose -f build/docker-compose.linting.yaml up
```

### 2.4. Visual Aids

- **Diagrams and Screenshots:** Enhance explanations with relevant diagrams, flowcharts, or screenshots. Visual aids can simplify complex information and provide clarity. Ensure all images are appropriately labeled and referenced in the text.

### 2.5. Consistency and Style

- **Uniform Formatting:** Adhere to a consistent formatting style throughout the document. This includes uniform use of fonts, text sizes, and code block styles. Consistency aids in maintaining a professional appearance and enhances readability.
- **Style Guide Compliance:** Follow an established style guide to maintain consistency in language, tone, and formatting. This ensures that all documentation aligns with the project's standards and provides a cohesive reading experience.

### 2.6. Maintenance

- **Regular Updates:** Keep the documentation up-to-date with any changes in the project. Regular reviews should be conducted to ensure accuracy and relevance.
- **Version Control:** Utilize version control systems to manage changes to the documentation. This allows for tracking revisions and collaborating effectively with other team members.
