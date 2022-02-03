# SemiF-AnnotationPipelinev

A data pipeline is a series of steps that takes raw data from different sources and moves the data to a destination for loading, transforming, and analysis.

A simple and straight forward DataLoader requires well structured data.

There should be a clear connection between images, image compenents and tabular data or metadata. Unique ids can be used to create a clear link between the variables in the table and one or multiple images.

A DataLoader loads samples of images and metadata and returns batches of tensors of a given size making training a lot faster and organizing code.