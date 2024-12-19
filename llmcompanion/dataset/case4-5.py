
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, MapType, ArrayType, IntegerType, BinaryType
from pyspark.sql.functions import udf

from utils.trec_car_tools import iter_pages, Para, Paragraph, ParaBody, ParaText, ParaLink, Section, Image, List

import protocol_buffers.page_pb2 as page_pb2
import pandas as pd
import pickle
import spacy
import time
import json
import os
import json


# define valid classes
SKELETON_CLASSES = (Para, List, Section, Image)
PARAGRAPH_CLASSES = (Paragraph)


def write_file_from_DataFrame(df, path, file_type='parquet'):
    """ Writes a PySpark DataFrame to different file formats """
    if file_type == 'parquet':
        df.write.parquet(path + '_' + str(time.time()))


def write_pages_data_to_dir(read_path, dir_path, num_pages=1, chunks=100000, print_intervals=100, write_output=False):
    """ Reads TREC CAR cbor file and returns list of Pages as bytearrays """

    # create new dir to store data chunks
    if (os.path.isdir(dir_path) == False) and write_output:
        print('making dir:'.format(dir_path))
        os.mkdir(dir_path)

    def write_to_parquet(data, dir_path, chunk):
        """ write data chunks to parquet """
        parquet_path = dir_path + 'page_data_chunk_' + str(chunk) + '.parquet'
        columns = ['idx', 'chunk', 'page_id', 'page_name', 'page_bytearray']
        pd.DataFrame(data, columns=columns).to_parquet(parquet_path)

    chunk = 0
    pages_data = []
    with open(read_path, 'rb') as f:
        t_start = time.time()
        for i, page in enumerate(iter_pages(f)):

            # stops when 'num_pages' processed
            if i >= num_pages:
                break

            # add bytearray of trec_car_tool.Page object
            #TODO - unpack dict here?
            pages_data.append([i, chunk, page.page_id, page.page_name, bytearray(pickle.dumps(page))])

            # write data chunk to file
            if ((i+1) % chunks == 0) and (i != 0 or num_pages == 1):
                if write_output:
                    print('WRITING TO FILE: {}'.format(i))
                    write_to_parquet(data=pages_data, dir_path=dir_path, chunk=chunk)

                    # begin new list
                    pages_data = []
                    chunk += 1

            # prints update at 'print_pages' intervals
            if (i % print_intervals == 0):
                print('----- STEP {} -----'.format(i))
                time_delta = time.time() - t_start
                print('time elapse: {} --> time / page: {}'.format(time_delta, time_delta / (i + 1)))

    if write_output and (len(pages_data) > 0):
        print('WRITING FINAL FILE: {}'.format(i))
        write_to_parquet(data=pages_data, dir_path=dir_path, chunk=chunk)

    time_delta = time.time() - t_start
    print('PROCESSED DATA: {} --> processing time / page: {}'.format(time_delta, time_delta / (i + 1)))


def pyspark_processing(dir_path):
    """ PySpark pipeline for adding syethetic entity linking and associated metadata """

    @udf(returnType=BinaryType())
    def synthetic_page_skeleton_and_paragraphs_udf(p):
        """ PySpark udf creating a new Page.skeleton with synthetic entity linking + paragraph list """

        def get_bodies_from_text(spacy_model, text):
            """ build list of trec_car_tools ParaText & ParaLink objects (i.e. bodies) from raw text """
            # nlp process text
            doc = spacy_model(text=text)
            # extract NED (named entity detection) features
            ned_data = [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents]

            text_i = 0
            text_end = len(text)
            new_text = ''
            bodies = []
            for span, start_i, end_i in ned_data:
                if text_i < start_i:
                    # add ParaText object to bodies list
                    current_span = text[text_i:start_i]
                    bodies.append(ParaText(text=current_span))
                    new_text += current_span

                # add ParaLink object to bodies list
                current_span = span
                new_text += current_span
                # TODO - entity linking
                bodies.append(ParaLink(page='STUB_PAGE',
                                       pageid='STUB_PAGE_ID',
                                       link_section=None,
                                       anchor_text=current_span))
                text_i = end_i

            if text_i < text_end:
                # add ParaText object to bodies list
                current_span = text[text_i:text_end]
                bodies.append(ParaText(text=current_span))
                new_text += current_span

            # assert appended current_span equal original text
            assert new_text == text, {"TEXT: {} \nNEW TEXT: {}"}

            return bodies


        def parse_skeleton_subclass(skeleton_subclass, spacy_model):
            """ parse PageSkeleton object {Para, Image, Section, Section} with new entity linking """

            if isinstance(skeleton_subclass, Para):
                para_id = skeleton_subclass.paragraph.para_id
                text = skeleton_subclass.paragraph.get_text()
                # add synthetic entity linking
                bodies = get_bodies_from_text(spacy_model=spacy_model, text=text)
                p = Paragraph(para_id=para_id, bodies=bodies)
                return Para(p), p

            elif isinstance(skeleton_subclass, Image):
                caption = skeleton_subclass.caption
                # TODO - what is a paragraph??
                s, p = parse_skeleton_subclass(skeleton_subclass=caption, spacy_model=spacy_model)
                imageurl = skeleton_subclass.imageurl
                return Image(imageurl=imageurl, caption=s), p

            elif isinstance(skeleton_subclass, Section):
                heading = skeleton_subclass.heading
                headingId = skeleton_subclass.headingId
                children = skeleton_subclass.children

                if len(children) == 0:
                    return Section(heading=heading, headingId=headingId, children=children), []

                else:
                    s_list = []
                    p_list = []
                    # loop over Section.children to add entity linking and re-configure to original dimensions
                    for c in children:
                        s, p = parse_skeleton_subclass(skeleton_subclass=c, spacy_model=spacy_model)
                        if isinstance(s, SKELETON_CLASSES):
                            s_list.append(s)
                        if isinstance(p, list):
                            for paragraph in p:
                                if isinstance(paragraph, PARAGRAPH_CLASSES):
                                    p_list.append(paragraph)
                        else:
                            if isinstance(p, PARAGRAPH_CLASSES):
                                p_list.append(p)
                    return Section(heading=heading, headingId=headingId, children=s_list), p_list

            elif isinstance(skeleton_subclass, List):
                level = skeleton_subclass.level
                para_id = skeleton_subclass.body.para_id
                text = skeleton_subclass.get_text()
                # add synthetic entity linking
                bodies = get_bodies_from_text(spacy_model=spacy_model, text=text)
                # TODO - what is a paragraph??
                p = Paragraph(para_id=para_id, bodies=bodies)
                return List(level=level, body=p), p

            else:
                raise ValueError("Not expected class")


        def parse_skeleton(skeleton, spacy_model):
            """ parse Page.skeleton (i.e. list of PageSkeleton objects) and add synthetic entity linking """

            synthetic_skeleton = []
            synthetic_paragraphs = []
            for i, skeleton_subclass in enumerate(skeleton):
                s, p = parse_skeleton_subclass(skeleton_subclass, spacy_model)
                if isinstance(s, SKELETON_CLASSES):
                    synthetic_skeleton.append(s)
                if isinstance(p, list):
                    for paragraph in p:
                        if isinstance(paragraph, PARAGRAPH_CLASSES):
                            synthetic_paragraphs.append(paragraph)
                else:
                    if isinstance(p, PARAGRAPH_CLASSES):
                        synthetic_paragraphs.append(p)

            return synthetic_skeleton, synthetic_paragraphs

        # initialise spacy_model
        spacy_model = spacy.load("en_core_web_lg")
        # extract skeleton (list of PageSkeleton objects)
        skeleton = pickle.loads(p).skeleton

        synthetic_skeleton, synthetic_paragraphs = parse_skeleton(skeleton=skeleton, spacy_model=spacy_model)

        return bytearray(pickle.dumps([synthetic_skeleton, synthetic_paragraphs]))


    # TODO -  sythetics_inlink_anchors
    # TODO - sythetics_inlink_ids
    # TODO - expose metadata?

    # add PySpark rows
    spark = SparkSession.builder.appName('trec_car_spark').getOrCreate()

    # creare pyspark DataFrame where each row in a bytearray of trec_car_tool.Page object
    df = spark.read.parquet(dir_path)

    print('START f.printSchema():')
    df.printSchema()

    df = df.withColumn("synthetic_entity_linking", synthetic_page_skeleton_and_paragraphs_udf("page_bytearray"))

    print('END df.printSchema():')
    df.printSchema()

    return df

def read_from_protobuf():
    """ """
    #TODO - desc
    # TODO - write
    with open(path, "rb") as f:
        print("read values")
        simple_message_read = page_pb2.PageMessage().FromString(f.read())

def write_to_protobuf(df, path, print_intervals=1000):
    t_start = time.time()
    with open(path, "wb") as f:
        for i, row in enumerate(df.rdd.collect()):
            page_message = page_pb2.PageMessage()
            page_message.idx = row[0]
            page_message.chunk = row[1]
            page_message.page_id = row[2]
            page_message.page_name = row[3]
            #TODO - double pickle punchy?
            page_message.page = pickle.dumps(pickle.loads(row[4]))
            page_message.synthetic_paragraphs = pickle.dumps(pickle.loads(row[5])[0])
            page_message.synthetic_skeleton = pickle.dumps(pickle.loads(row[5])[1])

            bytesAsString = page_message.SerializeToString()
            f.write(bytesAsString)

            if (i % print_intervals == 0):
                print("written row {} - page_id={} (time = {})".format(i, row[0], time.time()-t_start))

    print('FINISHED in {}'.format(time.time()-t_start))


def run_pyspark_job(read_path, dir_path, output_path, num_pages=1, chunks=100000, print_intervals=100,
                    write_output=False):
    # extract page data from
    write_pages_data_to_dir(read_path=read_path,
                            dir_path=dir_path,
                            num_pages=num_pages,
                            chunks=chunks,
                            print_intervals=print_intervals,
                            write_output=write_output)

    # process page data adding synthetic entity links
    df = pyspark_processing(dir_path=dir_path)

    write_to_protobuf(df=df, path=output_path, print_intervals=print_intervals)


if __name__ == '__main__':
    # read_path = '/nfs/trec_car/data/pages/unprocessedAllButBenchmark.Y2.cbor'
    read_path = '/nfs/trec_car/entity_processing/trec-car-entity-processing/data/test.pages.cbor'
    dir_path = '/nfs/trec_car/data/test_entity/data_{}/'.format(str(time.time()))
    output_path = '/nfs/trec_car/data/test_entity/output.bin'
    num_pages = 50
    write_output = True
    chunks = 5
    print_intervals = 5
    df = run_pyspark_job(read_path=read_path, dir_path=dir_path, num_pages=num_pages, chunks=chunks,
                         print_intervals=print_intervals, write_output=write_output, output_path=output_path)
