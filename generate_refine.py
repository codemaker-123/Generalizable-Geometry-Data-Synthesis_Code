import sys
import os
import json
import signal
import random
import multiprocessing
import time
from contextlib import contextmanager
import logging
from tqdm import tqdm  

sys.path.append('..')

from utils.loading_utils import load_definitions_and_rules
import graph as gh
import problem as pr
from clause_generation import *
from pretty_problem_statement_dict import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


need_proof = False
augment = False  
test = True
generate_num = 10000 if not test else 1000
process_num = max(1, multiprocessing.cpu_count() - 1) 
batch_size = 20  
stage2 = 2 * generate_num // 10  # 40% of total
stage3 = 6 * generate_num // 10  # 80% of total (meaning 20% remains after this point)
max_retries = 5  


if not test:
    save_dir = f'./dataset_reinforce_10K_hard_small_new/'
else:
    save_dir = './dataset_test_hard_small_new/'

logger.info(f"Saving to: {save_dir}")
logger.info(f"Using {process_num} processes")

os.makedirs(save_dir + "/img", exist_ok=True)
os.makedirs(save_dir + "/temp", exist_ok=True)  

random.seed(5 if not test else 7)

defs_path = './defs.txt'
rules_path = './rules.txt'
definitions, rules = load_definitions_and_rules(defs_path, rules_path)


lock = multiprocessing.Lock()
manager = multiprocessing.Manager()
generated_ids = manager.list()  
failed_ids = manager.list()     

uppercase_space_cache = {}

def cached_remove_uppercase_space(text):
    if text not in uppercase_space_cache:
        uppercase_space_cache[text] = remove_uppercase_space(text)
    return uppercase_space_cache[text]

def gen_data_batch(index_start, index_end, process_index, num_analysis, progress_dict):
    batch_entries = []
    i = index_start
    
    while i < index_end:
        start_time = time.time()
        retry_count = 0
        success = False
        
        progress_dict[process_index] = (i - index_start) / (index_end - index_start)
        
        logger.debug(f"Process {process_index} processing {i}/{index_end}")
        
        while retry_count < max_retries and not success:
            try:
                cc_gen = CompoundClauseGen(definitions, 1 if i < stage2 else (2 if i < stage3 else 3))
                txt = cc_gen.generate_clauses()
                p = pr.Problem.from_txt(txt)
                
                angles = []
                for cl in p.clauses:
                    for cons in cl.constructions:
                        if "s_angle" in str(cons):
                            p1 = cons.args[0]
                            head = cons.args[1]
                            p2 = cons.args[2]
                            angle = cons.args[3]
                            angles.append((p1, head, p2, angle))
                
                try:
                    with timeout(20):
                        g, _ = gh.Graph.build_problem(p, definitions)
                        para = [] 

                        for sublist in list(g.cache.keys()):
                            if 'para' in sublist:
                                para.append(sublist)
                        equ_angle = [[n.name for n in segment.equivs()] for segment in g.type2nodes[gh.Measure]]
                except TimeoutError:
                    logger.warning(f"Graph building timed out for index {i}, retrying ({retry_count+1}/{max_retries})...")
                    retry_count += 1
                    continue
                    
                temp_img_path = save_dir+f"/temp/{i}_{process_index}.jpg"
                final_img_path = save_dir+f"/img/{i}.jpg"
                
                try:
                    with timeout(20):
                        len_name_len = gh.nm.draw_reinforce(
                            g.type2nodes[gh.Point],
                            g.type2nodes[gh.Line],
                            g.type2nodes[gh.Circle],
                            g.type2nodes[gh.Length],
                            theme='',
                            angle=angles,
                            equ_angle=equ_angle,
                            para = para,
                            save_to=temp_img_path
                        )
                        
                        if not os.path.exists(temp_img_path) or os.path.getsize(temp_img_path) < 1000:  # 小于1KB可能是空图或错误图
                            raise Exception("Generated image is invalid or too small")
                            
                        if os.path.exists(temp_img_path):
                            import shutil
                            shutil.move(temp_img_path, final_img_path)
                            
                except TimeoutError:
                    logger.warning(f"Drawing timed out for index {i}, retrying ({retry_count+1}/{max_retries})...")
                    retry_count += 1
                    continue
                    
                filtered_data = [(length, name) for length, name in len_name_len if length is not None]
                if filtered_data:
                    # Generate the main sentence parts
                    sentences = [f"the length of {name} is {length:.2f}" for length, name in filtered_data]
                    
                    # Capitalize only the first sentence (without altering the name capitalization)
                    sentences[0] = sentences[0][0].upper() + sentences[0][1:]

                    # Join all but the last element with commas and then append 'and' before the last one
                    if len(sentences) > 1:
                        result = ", ".join(sentences[:-1]) + " and " + sentences[-1]
                    else:
                        result = sentences[0]
                    result = result + "."
                else:
                    result = ""
                # result = ".".join(sentences) + "." if sentences else None
                
                # if augment:
                #     img_path = save_dir+f"/img/{i}.jpg"
                #     if random.random() < 0.5:  # 只有50%的概率应用增强
                #         import cv2
                #         img = cv2.imread(img_path)
                #         img = occlude(img, occluder_ratio=0.25)
                #         cv2.imwrite(img_path, img)
                
                processed_clauses = [cached_remove_uppercase_space(clause_item) for clause_item in txt.split(";")]
                
                new_entry = {
                    "id": f"{i}",
                    "image": f"img/{i}.jpg",
                    "conversations": [
                        {"from": "human", "value": "Render a clear and concise description of an image about geometric shapes.\n<image>"},
                        {"from": "gpt", "value": gen_nl(txt) + (result if result is not None else "")}
                    ],
                    "clause": processed_clauses
                }
                
                batch_entries.append(new_entry)
                
                with lock:
                    generated_ids.append(i)
                
                success = True
                
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                retry_count += 1
                logger.error(f"Error occurred at index {i}: {str(e)}")
                if retry_count >= max_retries:
                    logger.error(f"Max retries reached for index {i}, skipping...")
                    with lock:
                        failed_ids.append(i)
                    success = True  
                time.sleep(0.1)  
        
        if len(batch_entries) >= batch_size or i == index_end - 1:
            if batch_entries: 
                with lock:
                    try:
                        tmp_file = f"{save_dir}/temp/batch_{process_index}_{i}.json"
                        with open(tmp_file, "w") as f:
                            json.dump(batch_entries, f, indent=2)
                        
                        with open(tmp_file, "r") as f:
                            json.load(f)  
                        
                        with open(f"{save_dir}/data.json", "a") as f:
                            for entry in batch_entries:
                                json.dump(entry, f)
                                f.write(",\n")
                                
                                for clause_item in entry["clause"]:
                                    num_analysis[clause_item] = num_analysis.get(clause_item, 0) + 1
                        
                        os.remove(tmp_file)
                    except Exception as e:
                        logger.error(f"Error writing batch data: {str(e)}")
            
            batch_entries = [] 
        
        i += 1
        process_time = time.time() - start_time
        logger.debug(f"Process {process_index} completed item {i-1} in {process_time:.2f} seconds")

def fill_missing_items(missing_ids, num_analysis):

    logger.info(f"Filling {len(missing_ids)} missing items...")
    for i in missing_ids:
        gen_data_batch(i, i+1, 0, num_analysis, {})

def main():

    num_analysis = manager.dict()
    progress_dict = manager.dict() 
    

    with open(f"{save_dir}/data.json", "w") as f:
        f.write("[\n")  
    

    pool = multiprocessing.Pool(process_num)

    tasks = []
    chunk_size = max(10, generate_num // (process_num * 5)) 
    
    for start_idx in range(0, generate_num, chunk_size):
        end_idx = min(start_idx + chunk_size, generate_num)
        process_id = len(tasks)
        progress_dict[process_id] = 0.0
        tasks.append((start_idx, end_idx, process_id, num_analysis, progress_dict))
    

    result = pool.starmap_async(gen_data_batch, tasks)
    

    with tqdm(total=100) as pbar:
        while not result.ready():

            total_progress = sum(progress_dict.values()) / len(progress_dict) * 100
            pbar.n = int(total_progress)
            pbar.refresh()
            time.sleep(1)
    
    try:
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        logger.error("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        pool.join()
        sys.exit(1)
    

    all_ids = set(range(generate_num))
    generated_set = set(generated_ids)
    missing_ids = all_ids - generated_set
    
    logger.info(f"Generation completed. Generated: {len(generated_set)}/{generate_num}")
    
    if missing_ids:
        logger.info(f"Found {len(missing_ids)} missing items. Attempting to fill them...")
        fill_missing_items(list(missing_ids), num_analysis)
    

    generated_set = set(generated_ids)
    missing_after_fill = all_ids - generated_set
    
    if missing_after_fill:
        logger.warning(f"After filling, still missing {len(missing_after_fill)} items: {missing_after_fill}")
    
    with open(f"{save_dir}/data.json", "a") as f:
        f.write("]\n") 
    
    with open(f"{save_dir}/data.json", "r") as f:
        content = f.read()
    content = content.rstrip(",\n") + "\n]\n"
    with open(f"{save_dir}/data.json", "w") as f:
        f.write(content)
    
    logger.info("Data generation completed.")
    
    sorted_data = sorted(num_analysis.items(), key=lambda x: x[1], reverse=True)
    with open(f"{save_dir}/sorted_data.json", "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=2)
    
    with open(f"{save_dir}/failed_ids.json", "w") as f:
        json.dump(list(failed_ids), f)
    
    logger.info(f"Total unique clauses: {len(sorted_data)}")
    logger.info(f"Successfully generated: {len(generated_ids)}/{generate_num} items")
    logger.info(f"Failed: {len(failed_ids)} items")

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")