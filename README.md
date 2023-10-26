# Reliable and Efficient Estimation of Knowledge Graph Accuracy
Data accuracy is a central dimension of data quality, especially when dealing with Knowledge Graphs (KGs). Auditing the accuracy of KGs is essential to make informed decisions in entity-oriented services or applications relying on KG embeddings, such as virtual assistants. 
However, manually evaluating the accuracy of large-scale KGs is prohibitively expensive. 

This work contributes to the research on developing efficient sampling techniques for estimating KG accuracy. To this end, we introduce a suite of methods building upon the Wilson method for confidence interval estimation, providing a more dependable alternative to the commonly used Wald method. We also propose solutions for adapting the Wilson method to complex sampling designs, making it applicable to various evaluation scenarios. 

The experiments show that the presented methods increase the reliability of accuracy estimates by up to four times when compared to state-of-the-art methods, while preserving or enhancing efficiency. Additionally, this consistency holds regardless of the KG size or structure.
