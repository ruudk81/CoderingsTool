Codebook generation for open ended survey questions by GATOS



\- GATOS = Generative AI-enabled Theme Organization and Structuring 



Step 1: Summarize the Original Data

The first step in the GATOS workflow is to summarize the original data into distinct ideas. This step is necessary because the raw data may contain multiple ideas in a single response.For example, a participant may write a long response that contains multiple ideas about their perspective of their organization’s culture of ethics. The goal of this step is to extract the key ideas from the raw data to make them easier to analyze in the subsequent steps. One can use a relatively small language model (e.g., 7 to 14 billion parameters) for this step. Here, we used the mistral-nemo-12b model for this information extraction task because it is small enough to run quickly but performant enough not to miss parts of the original data to summarize (along with being an open-source model and released under an Apache-2.0 license). The prompt used for summarization is given in the Appendix 8.2. An example of the input and output for this information extraction is shown in table 4.



Step 2: Clustering Semantically Similar Ideas 

The second step in the workflow is to identify semantically similar summary points from step one. The goal of this step is to group together the instances when the same idea is expressed in different ways in the data. If we can do this grouping well, then we should be able to proceed through the clusters one-at-a-time to identify recurring patterns more easily than if there were several discrete ideas covered in each cluster. To accomplish step 2, we first embed all of the summary ideas using a text embedding model. In our case here we used the ‘mxbai’ model \[37] because it is a lightweight performant model according to the MTEB leaderboard \[38] released under an Apache-2.0 license. The next step is to reduce the dimensionality of these 1,024-dimensional embeddings because clustering in this igh-dimensional space might suffer from the curse of dimensionality \[39], where all points tend to be far from each other in high-dimensional spaces. We use principal component analysis (PCA) plus uniform manifold approximation and projection (UMAP) \[40] to reduce the dimensionality of the embeddings to a lower-dimensional space. We first use PCA to reduce from 1,024 dimensions to D dimensions, where D is dynamically found by identifying whichever dimension number retains 90% of the variance in the data. In practice, we tended to find 100 ≤ D ≤ 120. From this intermediate embedding space, we then used UMAP to reduce down to five dimensions. We use UMAP here to try and preserve the global structure of the data manifold and improve clustering results. From this low-dimensional space, we then use agglomerative clustering with Euclidean distance to cluster the summary points. We use agglomerative clustering rather than K-means because our initial testing suggested 14 that more homogeneous clusters were identified with agglomerative clustering. The output of this step is a set of clusters of semantically similar summary points.



Step 3.1: Create Set of Speculative Starter Codes 

The next step in the GATOS workflow is to create the actual codebook by iteratively reading the clusters of summary points and deciding whether to generate a new code or not. To start the process, however, we prompt a generative text model to create 20 hypothetical codes that one might expect to appear in a study of whatever topic we simulated. For example, for the teammate feedback study, we prompted the model to generate 20 hypothetical codes one

would expect to describe some data collected in a study of teammate feedback. These initial synthetic codes are used to help the process get started by providing the process with some initial codes to consider from the nearest neighbor matching when generating new codes. The specific prompt is given in the Appendix 8.3.



Step 3.2: Inductive Codebook Generation

Step 3.2.1 Preprocess phase

With the clusters of summary points and speculative starter codes generated, we can now begin the iterative inductive codebook generation. We accomplish this by embedding each summary point that belonged to a cluster. We once again used the mxbai embedding model as in Step 2. We then embed all entries in the codebook (at the start, this is 20 codes, but as the codebook grows this number also grows). After embedding the extracted information and the codes from the growing codebook, we find the k nearest neighbors codes for each of the extracted ideas in the cluster. We use cosine similarity for this matching. Cosine similarity is a measure of similarity between two non-zero vectors belonging to an inner product space. The similarity score is defined to equal the cosine of the angle between them, which is also the same as the inner product of the same vectors when normalized to both have unit length. In plain language, cosine similarity measures similarity between two vectors, e.g., embedding vectors. We use the cosine similarity to find the k nearest neighbors for each of the extracted ideas. In the present study, we set k = 5. We chose 5 because to balance between having enough potentially relevant codes in the codebook for the generative text model to check without giving the model all of the codes. Giving the model all of the codes might be a bad idea because it could distract or overwhelm the model’s attention. The output of this neighbor matching step is a set of k nearest neighbors for each of the extracted ideas. We use these as part of the prompt to the generative text model to decide whether to generate a new code or not based on the extracted idea and the extant codes in the codebook. 



Simply put, the procedure is as follows. First, embed the new cluster of summary points that the model is going to analyze. Next, embed all the  entries in the codebook. Then, calculate cosine similarity between the embedding for the new text and the codes in the codebook to find the k nearest codes in the codebook for the new piece of text. In our case, we set k as 5. Remove duplicate codes from this set. Finally, include only this small subset of the codes inthe prompt to the generative text model when instructing it to consider whether or not to 15 generate a new code based on the new cluster it is reading and the subset of the codebook it can reference.



Step 3.2.2: Generation phase - i.e. Using Generative Text Model to Decide Whether to Generate a New Code

The next phase in our process runs C many times, where C is the number of clusters from step two. As mentioned before, at a high level we instruct the generative text model to read the summary points in a single cluster, read the nearest neighbor codes from the codebook that might describe the ideas in that cluster, and then decide whether to generate a new code or not. We used Mistral-22b-2409 for this step because it is sufficiently large to be able

to exhibit reasoning steps while being small enough to run quickly on common consumer hardware. The prompt used for this step is given in the Appendix 8.4.



There are six specific steps that the model is instructed to complete as part of this decisionmaking process. First, the model is instructed to read the existing codebook. Second, the model is instructed to read the summary data points in cluster. Next, the model is instructed to try to use one or more of the existing codes to describe the summaries in the cluster. Fourth, the model is instructed to create a new code if needed. The model then must evaluate whether the suggested new code adheres to three evaluation criteria: parsimony, abstraction level, and non-redundancy. Finally, the model is instructed to make a final recommendation about whether to create a new code or not.



This workflow sounds complicated, but the actual philosophy is similar to what a traditional qualitative research might do when creating a codebook inductively: start reading data, create a code, and add that to the codebook. Then, when reading the next piece of data, consider the existing codes and see if the new data can be described by one of those existing codes in the codebook. If the new data point already has a code that describes it, then there

is no need to generate a new code, so proceed to the next piece of data. On the other hand, if the new data does not already have a code that describe it, then create a new code. We repeat this process for each of the n pieces of data (i.e., text) in the dataset.



Step 4: Codebook Simplification Through Theme Identification

The final step in the method is to simplify the codebook by trying to identify themes. This step is necessary because the generative text model may generate redundant codes in the preceding step and codes that belong together at a more abstract level. To address these issues, we used a step in which we clustered similar codes together and gave those to the model along with the instructions to identify higher-level themes in the clusters of codes where possible. The specific prompt used in this step is given in the Appendix 8.5. Th  final output of the GATOS workflow is a set of themes and codes that should describe the common semantic patterns in the data.





Appendix 



8.1 Prompts

This section contains the prompts used for the steps in the GATOS workflow.



8.2 Initial Summarization Prompt

Information Extraction Instructions



"You are an expert text analyst reading {data type}s collected in {data collection context}. I

am going to send you one of these {data type}s. I need you to use your expertise to analyze

the provided text in the <text> tag below and summarize it in an enumerated list. You

should do this analysis by providing several short descriptive phrases that summarize each

idea discussed in the {data type} that answered the prompt. When you suggest multiple items

, separate each one in your response with a new line. You MUST remove anyone’s names and

∗use gender neutral pronouns∗ for deidentification purposes. Start your response with ‘‘My

summary:”. Here is an example of input and desired output from a different context when

there are only two topics, but remember that you can suggest as many topics as you think are

necessary for the text you summarize.

Example input: ‘‘Jared did a great job responding quickly to emails and turning in good work

.’’

Example output: ‘‘My summary:

1\. Responded quickly to emails

2\. Turned in good work’’.

Notice how the main ideas are summarized and there are no names or pronouns included here.

Also, notice how the response did not make up information that was not in the input. You

must NEVER make up information that is not in the input text you receive because there is a

severe penalty for that. If the text you receive is very short and says ‘‘nothing’’, do not make

up new things.

Here is the text for you to summarize: <text>{text}</text>

Begin your analysis now."



8.3 Initial Codebook Creation Prompt

Initial Codebook Creation



"Act as if you are the world’s best qualitative data analysis. You specialize in applying codes to

analyze qualitative data. I need your help. Your important task is to generate {k to start}

hypothetical codes that one might encounter when analyzing {data type}s from {

data collection context}. You should format your response by filling in the template I give you

at the end of these instructions, which is an enumerated list of {k to start} codes. The list

should contain {k to start} short phrases with regular spacing between words written in plain

English without examples. After the final code, you should stop writing so that it is easy for

your response to be parsed for downstream tasks. Begin your list now using the following

template:\\n{code template}"



where code template is a numbered string of k to start codes. If k to start is 5, then the code template would be 1. Code 1 2. Code 2 3. Code 3 4. Code 4 5. Code 5



8.4 Inductive Codebook Generation Prompt

The inductive codebook generation prompt contains multiple parts. The first part introduces the persona for the model to adopt for this task. Prior research suggests persona assignment can be a way to improve generative text model performance \[57, 58]. The next part of this initial portion of the prompt provides the model with the background information about the task.



Inductive Codebook Generation

"Act as if you are the world’s best qualitative data analyst with expertise in generating

qualitative codebooks for thematic analysis. You specialize in creating parsimonious

codebooks with non−overlapping and non−redundant codes. A codebook in this setting is a

collection of labels and definitions for those labels that can be used to describe pieces of data

in a qualitative research study. I need your help to create a qualitative codebook to analyze {

data type}s from {data collection context}. To aid you in this process, I am going to send you

instructions in the <instructions> XML tag. Use the instructions to analyze the data in the

<data to analyze> tag. You must follow these instructions using your expertise and data to

analyze in the <data to analyze> XML tag. I will provide you the instructions first and then

the data to analyze afterward. Be aware that your instructions contain task instructions,

evaluation criteria, and formatting instructions, each in their respective XML tags."



The next portion of the prompt introduces the task instructions for the model to follow.



Instructions for Inductive Codebook Generation

"<instructions>

<task instructions>

We are trying to determine whether or not an exsiting codebook is sufficient for analyzing one

{data type} that you have been given in the <text to analyze> tag. Your important task is

to analyze one summary of {data type}s collected in the context of {data collection context}

and determine if the theme discussed in the {data type} summary is already covered by the

codes in an existing codebook that will be given to you in the <existing codebook> tag or if

instead the codebook needs one or more new code to cover the theme in the text to analyze.

You should complete your task by following these steps:"



Steps 1-4 for Inductive Codebook Generation

" theme discussed in the summary.

Step 3: Try to use existing codebook.

Attempt to describe the main theme of the {data type} using one or more of the existing

codes in the existing codebook. Think at a high level of abstraction and consider if any new

themes could be subcategories of existing codes. If you determine that there is no need to

create a new code, say ”No new codes needed”.

Step 4: Create new code if needed.

If in step 3 you discover that you are unable to use the current codes to describe the main

theme in the summary of the {data type} that you are analyzing, determine whether the

existing codebook needs new labels to describe the summary in the <text to analyze> tag.

You should complete this determination by reasoning step−by−step. If you determine that a

new code is necessary, explicitly justify why existing codes or combinations thereof are

insufficient. Finally, generate a new code (or codes, if multiple ones are absolutely necessary)

that captures the main concepts or themes discussed in the {data type}s that you review.

Remember, you specialize in creating parsimonious codebooks and avoid creating redundant

codes. Your goal is to use the least number of new codes possible while still accurately

representing the data.

There is a VERY significant penalty for creating redundant or unnecessary codes, so you

should only create a new code if you are ∗∗absolutely∗∗ certain the existing ones are

insufficient, even when combined or broadened. If you decide to generate a new code, please

provide:

− The code (a short phrase).

− A brief definition of what the label represents."



Steps 5-6 for Inductive Codebook Generation

"Step 5: Evaluate your suggestion.

To guide your work, you must consider the following three evaluation criteria. These three

evaluation criteria will be used by other famous expert qualitative data analysts to evaluate

the quality of your work. In the reflection step, you must check whether you have satisfied

each of these three criteria:

<evaluation criteria>

Evaluation Criteria 1. Parsimony: Have you made every effort to use existing codes or

combinations of existing codes before proposing a new one?

Evaluation Criteria 2. Abstraction Level: Is any proposed new code at an appropriate level of

abstraction, consistent with existing codes?

Evaluation Criteria 3. Non−Redundancy: Have you avoided creating codes that significantly

overlap with existing ones?

To help illustrate what I mean by non−redundancy, here is an example of redundant codes

and an explanation of their redundancy:

{redundancy example}

Use the evaluation criteria and these task instructions to help you in your step−by−step

reasoning for each of the preparation, analysis, and reflection steps given to you in these

instructions.

It is CRUCIAL TO REMEMBER that if you do not think a new code should be created, you

must say ”No new codes needed”.

</evaluation criteria>

Step 6: Final recommendation.

Present your final logical recommendation on a new line about any codes to create or whether

none are needed on a new line.

</task instructions>"



Formatting Instructions for Inductive Codebook Generation

" <formatting instructions>

I will give you a template to use for your response. The main parts of the template are the

following. First, your response should start with ”My expert analysis:”. Then, on a new line,

you should write your logical step−by−step reasoning about the existing codes and the {

data type}s. This will include the two orientation steps, the two analysis steps, the reflection

step, and the recommendation step. Your anlayis notes should be succinct and formatted in a

numbered list rather than long prose. This means that each step in your step−by−step

reasoning should get its own line as if it were a premise in a proof. These notes should be

logical, adhere perfectly to your task instructions, be concise, and be in a numbered list. Then

, on another new line, you should state ”My logical recommendation:” followed by your

recommendation on yet another new line. Your recommendations can either be ”No new codes

needed” if no new codes are needed or the actual codes you suggest adding to the codebook.

If you do think one or more new codes should be created, your response should start ’Code: ’

followed by your code, then on a new line ’Definition: ’ followed by your definition for that

code.

For example:

Code: <code 1>

Definition: <definition 1>

</formatting instructions>

This concludes your task and formatting instructions.

</instructions>"



Presenting Data and Template for Inductive Codebook Generation

"analyze>

<existing codebook>

{codes}

</existing codebook>

And here is a summary of one {data type} for you to analyze.

<text to analyze>

{text}

</text to analyze>

</data to analyze>

Now that you have meticulously studied the data to analyze using your task instructions,

formatting instructions, and evaluation criteria, take a moment to gather your expert thoughts

and observations. When you are ready, begin your flawless and logical step−by−step analysis

using the instructions and evaluation criteria outlined above. Be sure to display your

expertise in creating parsimonious codebooks and minimizing redundancy and use the full

analysis template, provided below. Be sure to use spaces in any codes you write rather than

concatenating words together (e.g., say ”example code” rather than ”examplecode”). Here is

the template to use for your analysis. Begin your expert analysis when you are ready.

FULL ANALYSIS TEMPLATE:

My expert analysis:

Step 1 (codebook examination)

\[your step 1 notes describing the existing code go here]

Step 2 (current data examination)

\[your step 2 notes go here to identify the main theme in the {data type}]

Step 3 (analysis part 1)

\[your step 3 notes to describe main theme in the {data type}s with existing codes here]

Step 4 (analysis part 2)

\[your step 4 notes considering whether to create new code here, favoring parsimony and

avoiding unnecessary code creation]

Step 5 (reflection on planned suggestions)

\[your evaluation reflection notes here reviewing the evaluation criteria]

My logical recommendation:

\[logical recommendation based on expert step−by−step reasoning about whether or not to

create zero, one, or more than one new codes. These notes will reflect your reputation for only

creating essential codes]"



8.5 Theme Identification Prompt

Theme Identification Instructions

"You are an expert qualitative researcher specializing in thematic analysis. Your task is

to analyze a list of codes that will be given to you below in the <codes> tag and

identify potential themes following the guidance of Braun and Clarke. The goal is to

identify themes that help to answer the research question ‘‘{research question}’’.

Please follow these steps outlined in the <instructions> tag carefully."



The specific instructions for the theme identification task are provided in the following box.

"<instructions>

Step 1. Review the list of codes provided below in the <codes> tag below. These

codes are being used to analyze {data type}s from {data collection context}.

Step 2. Look for patterns and shared meanings among the codes. Consider how

different codes might be combined based on underlying concepts or features of the

data.

Step 3. Identify overarching narratives that might represent broader themes or sub−

themes.

Step 4. Remember that themes don’t simply ”emerge” from the data. Actively

construe relationships among the codes and examine how these relationships inform

potential themes.

Step 5. Consider the importance and salience of potential themes. Remember, the

number of codes supporting a theme is less important than whether the pattern

communicates something meaningful that helps answer the research question(s). On

that note, remember that the research question for this research is {research question

}.

Step 6. Aim for themes that are distinctive yet coherent with the overall analysis.

Themes may even be contradictory to each other.

Step 7. Be willing to let go of codes or potential themes that don’t fit the overall

analysis. Consider creating a ”miscellaneous” category for codes that don’t fit

elsewhere.

Step 8. Strive for a balance in the number of themes − not so many that the analysis

becomes unwieldy, but enough to fully explore the depth and breadth of the data.

Step 9. For each theme, prepare a structured description including the theme name,

its underlying concept, associated codes, and how these codes relate to each other and

the overall theme.

Step 10. Reflect on your analysis considering: themes that seem too broad or narrow,

contradictions or unexpected patterns, need for subthemes, and codes that don’t fit

well into the current themes.

Step 11. Organize your analysis into a structured format with initial observations, an

array of suggested themes (each as an object with name, concept, codes, and

relationship), and your reflection.

</instructions>"



The prompt also includes a section for the model to provide the list of codes to analyze



Presenting Data and Template for Theme Identification

"Now that you have studied your instructions carefully, here is the list of codes to

analyze to identify themes related to the research question ”{research question}”:

<codes>

{labels}

</codes>

Proceed with your expert analysis, explaining your reasoning at each step. Present

your analysis in JSON format with the following structure:

{{

”initial observations”: \[

”observation1”

],

”suggested themes”: \[

{{

”theme name”: ”Theme 1”,

”concept”: ”Brief description of the underlying concept or narrative”,

”codes”: \[

”Code 1”

],

”relationship”: ”Brief explanation of how these codes relate to each other and

the overall theme”

}}

],

”reflection”: {{

”broad or narrow themes”: ”Discussion of any themes that seem too broad or too

narrow”,

”contradictions or unexpected patterns”: ”Description of any contradictions or

unexpected patterns”,

”potential subthemes”: ”Discussion of any need for subthemes within the main

themes”,

”unclassified codes”: ”List of any codes that were not included in the proposed

themes”

}}

}}

Use this JSON structure I have given you as a template. Expand on the template by

adding as many observations, themes, and codes as necessary based on your analysis.

Ensure that your response remains a valid JSON object. Do not include any text

outside of this JSON structure.

Now that you have thoroughly read your task instructions, formatting instructions,

and the codes to analyze, take a moment to gather your expert thoughts. Begin your

analysis when you are ready."



&nbsp;









