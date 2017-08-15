import org.apache.spark.SparkContext;
import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.DataFrameWriter;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.List;

public class SparkExperiment {

    public static void main(String ... args){
        SparkSession sparkSession = SparkSession.builder().appName("My App").getOrCreate();
 

        Dataset<String> bbcCorpus = sparkSession.read().textFile("/home/banqo/Documents/bbc/*");

        Tokenizer tokenizer = new Tokenizer().setInputCol("value").setOutputCol("tokens");

        Dataset<Row> tokenizedData = tokenizer.transform(bbcCorpus);
//        DataFrameWriter<Row> dataFrameWriter = new DataFrameWriter<>(tokenizedData);
//        dataFrameWriter.json("/home/banqo/Documents/test");

        //Initialize StopWordsRemover
        StopWordsRemover stopWordsRemover = new StopWordsRemover().setInputCol("tokens").setOutputCol("filtered");

        //Remove stop words
        Dataset<Row> filtered = stopWordsRemover.transform(tokenizedData);
        
        

        HashingTF hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("termFrequency").setNumFeatures(2000);

        Dataset<Row> withTermFrequencies = hashingTF.transform(filtered);

        IDFModel idfModel = new IDF().setInputCol("termFrequency").setOutputCol("features").fit(withTermFrequencies);

        Dataset<Row> rescaledData = idfModel.transform(withTermFrequencies);

        rescaledData.show(5);


        LDA lda = new LDA().setK(10).setMaxIter(10);
        LDAModel model = lda.fit(rescaledData);

        double ll = model.logLikelihood(rescaledData);
        double lp = model.logPerplexity(rescaledData);
        System.out.println("The lower bound on the log likelihood of the entire corpus: " + ll);
        System.out.println("The upper bound on perplexity: " + lp);

//// Describe topics.
        Dataset<Row> topics = model.describeTopics(5);
        System.out.println("The topics described by their top-weighted terms:");
        topics.show(5);
//
//// Shows the result.
        Dataset<Row> transformed = model.transform(rescaledData);
        transformed.show(5);

        sparkSession.stop();
    }
}

