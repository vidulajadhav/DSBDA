//  HADOOP PRACTICAL - MapReduce WordCount
//  File    : WordCount.java
//  Goal    : Count occurrences of each word in input file
//  Setup   : Local Standalone Hadoop

// HOW TO RUN ON UBUNTU TERMINAL (Hadoop must be installed):
// STEP 1 - Create input directory and input file
//   mkdir -p ~/wordcount/input
//   echo "Hello World Hello Hadoop is great Hadoop" > ~/wordcount/input/data.txt
//
// STEP 2 - Compile the Java file
//   hadoop com.sun.tools.javac.Main WordCount.java
//   jar cf WordCount.jar WordCount*.class
//
// STEP 3 - Run the MapReduce Job on Hadoop
//   hadoop jar WordCount.jar WordCount ~/wordcount/input ~/wordcount/output
//
// STEP 4 - See the output
//   hadoop fs -cat ~/wordcount/output/part-r-00000
//
// EXPECTED OUTPUT (for above input):
//   Hadoop    2
//   Hello     2
//   World     1
//   great     1
//   is        1
//
// NOTE: If output folder already exists, delete it first:
//   hadoop fs -rm -r ~/wordcount/output
//   OR use local delete:
//   rm -rf ~/wordcount/output

import java.io.IOException;
import java.util.*;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class WordCount {

    //MAPPER CLASS 
    // Input  : <LongWritable key, Text value>
    //          key   = byte offset of line in file (ignored)
    //          value = one line of text from input file
    // Output : <Text word, IntWritable 1>
    //          emits every word with count 1
    public static class Map
           extends Mapper<LongWritable, Text, Text, IntWritable> {

        private final static IntWritable one  = new IntWritable(1);
        private              Text        word = new Text();

        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            // Convert line to String
            String line = value.toString();

            // Tokenize — splits by spaces/tabs/newlines
            StringTokenizer tokenizer = new StringTokenizer(line);

            while (tokenizer.hasMoreTokens()) {
                // Set current token as word
                word.set(tokenizer.nextToken());
                // Emit (word, 1) pair
                context.write(word, one);
            }
        }
    }

    // REDUCER CLASS 
    // Input  : <Text word, Iterable<IntWritable> counts>
    //          All 1s for the same word are grouped together
    //          e.g. ("Hello", [1, 1]) for Hello appearing twice
    // Output : <Text word, IntWritable totalCount>
    //          emits word with its total count
    public static class Reduce
           extends Reducer<Text, IntWritable, Text, IntWritable> {

        public void reduce(Text key,
                           Iterable<IntWritable> values,
                           Context context)
                throws IOException, InterruptedException {

            int sum = 0;

            // Add all 1s together to get total count
            for (IntWritable val : values) {
                sum += val.get();
            }

            // Emit (word, totalCount)
            context.write(key, new IntWritable(sum));
        }
    }

    //MAIN 
    // Configures and launches the MapReduce Job
    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();

        // Create job with name "wordcount"
        Job job = new Job(conf, "wordcount");

        // Set output key and value types
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        // Set Mapper and Reducer classes
        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);

        // Set input and output formats
        // TextInputFormat  → reads input line by line
        // TextOutputFormat → writes output as plain text
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        // Set input and output paths from command line args
        // args[0] = input path
        // args[1] = output path
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        // Submit job and wait for completion
        job.waitForCompletion(true);
    }
}

// Terminal Commands Step by Step
// # STEP 1 — Create input folder and file
// mkdir -p ~/wordcount/input
// echo "Hello World Hello Hadoop is great Hadoop" > ~/wordcount/input/data.txt

// # STEP 2 — Compile Java file using Hadoop's compiler
// hadoop com.sun.tools.javac.Main WordCount.java

// # STEP 3 — Package compiled .class files into a JAR
// jar cf WordCount.jar WordCount*.class

// # STEP 4 — Run the MapReduce job
// # Format: hadoop jar <jarfile> <classname> <input> <output>
// hadoop jar WordCount.jar WordCount ~/wordcount/input ~/wordcount/output

// # STEP 5 — View the output
// hadoop fs -cat ~/wordcount/output/part-r-00000

// # IF OUTPUT FOLDER ALREADY EXISTS — delete it first
// rm -rf ~/wordcount/output
