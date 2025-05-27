import argparse
import string
import json
import logging
import os
import numpy as np

from tqdm import tqdm
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
from factscore.clm import CLM
from factscore.npm import NPM
from factscore.openai_lm import OpenAIModel
from factscore.Llama3LM import Llama3LM
from factscore.HFmodel import HFmodel
from factscore.retrieval import DocDB, Retrieval

class FactScorer(object):
    def __init__(self,
                 afv_model="meta-llama/Llama-3.1-8B-Instruct",
                 afg_model="meta-llama/Llama-3.1-8B-Instruct",
                 is_npm=True,
                 is_retrieval=True,
                 data_dir=".cache/factscore",
                 model_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 openai_key="api.key",
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 batch_size=256):
        self.afg_model = afg_model
        self.afv_model = afv_model
        self.is_npm = is_npm
        self.is_retrieval = is_retrieval
#        assert model_name in ["retrieval+inst-llama", "retrieval+inst-llama+npm", "retrieval+ChatGPT",
#                "npm", "retrieval+ChatGPT+npm", "retrieval+llama31+npm","retrieval+llama31" ]
        self.model_name = self.generate_config_name()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        self.openai_key = openai_key
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.cost_estimate = cost_estimate
        if "inst-llama" in self.model_name:
            self.lm = CLM("inst-llama-7B",
                          model_dir=os.path.join(model_dir, "inst-llama-7B"),
                          cache_file=os.path.join(cache_dir, "inst-llama-7B.pkl"))
        elif "ChatGPT" in self.model_name:
            self.lm = OpenAIModel("ChatGPT",
                                  cache_file=os.path.join(cache_dir, "ChatGPT.pkl"),
                                  key_path=openai_key)
        elif "Llama-3.1" in self.afv_model:
            self.lm = Llama3LM(self.afv_model,
                                cache_file=os.path.join(cache_dir, self.model_name))
        else:
            self.lm = HFmodel(self.afv_model,
                                cache_file=os.path.join(cache_dir, self.model_name))
        self.logger.debug("%s",self.model_name)

    def generate_config_name(self):
        afg_name = self.afg_model.split('/')[-1]
        afv_name = self.afv_model.split('/')[-1]
        model_name = [afg_name, afv_name]
        if self.is_npm:
            model_name.append("npm")
        if self.is_retrieval:
            model_name.insert(0,"retrieval")
        model_name = "+".join(model_name)
        return model_name

    def save_cache(self):
        if self.lm:
            self.lm.save_cache()
        if "npm" in self.model_name:
            for k, v in self.npm.items():
                v.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)
        if "npm" in self.model_name:
            cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
            self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
                                 "npm-single",
                                 cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))


    def print_cost_estimates(self, total_words, task, model):
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Number of tokens are roughly 4/3 of the number of words
        total_tokens = total_words * 4.0 / 3

        # https://openai.com/pricing
        # if we use davinci-003, the cost is $0.02 per 1000 tokens
        # if we use gpt-3.5-turbo, the cost is $0.002 per 1000 tokens
        # Davinci-003 discontinued
        if model == "davinci-003":
            rate = 0.02
        elif model == "gpt-3.5-turbo":
            rate = 0.0015

        total_cost = total_tokens * rate / 1000

        # print the total words, tokens, and cost along with rate
        logging.critical("""Estimated OpenAI API cost for %s ($%.3f per 1000 tokens):
        $%.2f for %d words and %d tokens",task, rate, total_cost, total_words, total_tokens""")

    def get_score(self,
                  topics,
                  generations,
                  gamma=10,
                  atomic_facts=None,
                  knowledge_source=None,
                  verbose=False):
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        if type(topics)==type(generations)==str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"

        ## I can provide the Atomic Facts myself and by pass the AF generation if I want to test 
        ## the evaluation
        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else: #Atomic FactGeneration
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator(model_name=self.afg_model,
                                                        demon_dir=os.path.join(self.data_dir, "demos"),
                                                        key_path=self.openai_key,
                                                        af_cache_file=os.path.join(self.cache_dir, "InstructGPT.pkl"))

            # estimate the total cost of atomic fact generation
            if "ChatGPT" in self.model_name:
                total_words = 0
                for gen in generations:
                    total_words += self.af_generator.run(gen, cost_estimate=self.cost_estimate)

                self.print_cost_estimates(total_words, task="atomic fact generation", model="davinci-003")

            if verbose:
                topics = tqdm(topics)

            ## Start obtaining Atomic Facts for each generation 
            atomic_facts = []
            for topic, gen in zip(topics, generations):
                # optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained:
                    atomic_facts.append(None)
                    continue
                # continue only when the response is not abstained
                curr_afs, _ = self.af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 10 == 0:
                    self.af_generator.save_cache()

            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()
            self.af_generator.lm.unload_model()

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        if "ChatGPT" in self.model_name:
            # estimate the total cost of response generation
            total_words = 0
            for topic, generation, facts in zip(topics, generations, atomic_facts):
                if facts is not None:
                    total_words += self._get_score(topic, generation, facts, knowledge_source, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-3.5-turbo")

        if verbose:
            topics = tqdm(topics)

        scores = []
        init_scores = []
        decisions = []
        # 
        for topic, generation, facts in zip(topics, generations, atomic_facts):
            if facts is None:
                decisions.append(None)
            else:
                decision = self._get_score(topic, generation, facts, knowledge_source)
                # Score is the average number of "is_supported" for generation
                score = np.mean([d["is_supported"] for d in decision])
                
                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/len(facts))
                    score = penalty * score
                
                decisions.append(decision)
                scores.append(score)
                if len(scores) % 10 == 0:
                    self.save_cache()

        self.save_cache()

        out = {"score": np.mean(scores),
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None]),
               "config_name": self.model_name}

        if gamma:
            out["init_score"] = np.mean(init_scores)
        
        return out

    def _get_score(self, topic, generation, atomic_facts, knowledge_source, cost_estimate=None):
        decisions = []
        total_words = 0
        # Prompt Construction
        for atom in atomic_facts:
            atom = atom.strip()
            if self.lm:
                if isinstance(topic, str):
                    topic = [topic]
                passages = []
                for tp in topic:
                    if len(passages) > 10:
                        continue
                    # change K to 3 for ASQA, original 5
                    passages.extend(self.retrieval[knowledge_source].get_passages(tp, atom, k=5))

                definition = "Answer the question about {} based on the given context.\n\n".format(
                    " ".join(topic)
                    )
                context = ""
                for psg_idx, psg in enumerate(reversed(passages)):
                    context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                definition += context.strip()
                self.logger.debug(f"definition: {definition}\ndefinition-1: {definition[-1]}")
                if definition[-1] not in string.punctuation:
                    definition += "."
                prompt = "{}\n\nInput: {} True or False?\nAnswer:".format(definition.strip(), atom.strip())

                if cost_estimate:
                    if cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                        total_words += len(prompt.split())
                    elif cost_estimate == "ignore_cache":
                        total_words += len(prompt.split())
                    continue

                output = self.lm.generate(prompt)

                if isinstance(output[1], np.ndarray) and self.lm.logits:
                    # TODO: assert with tokenizer vocab len
                    logits = np.array(output[1])
                    # when logits are available,
                    true_score = logits[self.lm.true_id]
                    false_score = logits[self.lm.false_id]
                    is_supported = true_score > false_score
                    is_supported = is_supported.item()
                    self.logger.debug("-------------------")
                    self.logger.debug(f"Prompt: {prompt}")
                    self.logger.debug(f'\nLogits:\nTrue: {true_score}\nFalse: {false_score}\nis_supported: {is_supported}')
                    self.logger.debug(f'Output: {output[0]}')
                    self.logger.debug("-------------------")
                else:
                    # when logits are unavailable
                    generated_answer = output[0].lower()
                    if "true" in generated_answer or "false" in generated_answer:
                        if "true" in generated_answer and "false" not in generated_answer:
                            is_supported = True
                        elif "false" in generated_answer and "true" not in generated_answer:
                            is_supported = False
                        else:
                            is_supported = generated_answer.index("true") > generated_answer.index("false")
                    else:
                        is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

            else:
                is_supported = True

            if is_supported and "npm" in self.model_name:
                if isinstance(topic, str):
                    npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
                    is_supported = npprob > 0.3
                

            decisions.append({"atom": atom, "is_supported": is_supported})

        if cost_estimate:
            return total_words
        if not isinstance(decisions, list):
            self.logger.debug("Smth wrong with %s", topic)
        return decisions

def convert_to_serializable(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):  # If arrays are involved
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute FactScore for generated outputs.")

    # Required arguments
    parser.add_argument('--input_path', type=str, required=True,
                        help="Path to the input JSONL file containing topics and generations.")

    # Model configuration arguments
    parser.add_argument('--afv_model', type=str, default="Llama-3.1-8B-Instruct",
                        help="Name of the Atomic Fact Verification model.")
    parser.add_argument('--afg_model', type=str, default="Llama-3.1-8B-Instruct",
                        help="Name of the Atomic Fact Generation model.")
    parser.add_argument('--is_npm', action='store_false',
                        help="Flag to enable Neural Probabilistic Model (NPM).")
    parser.add_argument('--is_retrieval', action='store_false',
                        help="Flag to enable retrieval-based scoring.")

    # Directories and paths
    parser.add_argument('--data_dir', type=str, default=".cache/factscore",
                        help="Directory to store data files.")
    parser.add_argument('--model_dir', type=str, default=".cache/factscore",
                        help="Directory to store model files.")
    parser.add_argument('--cache_dir', type=str, default=".cache/factscore",
                        help="Directory to store cache files.")
    parser.add_argument('--openai_key', type=str, default="api.key",
                        help="Path to the OpenAI API key file.")

    # Evaluation configuration
    parser.add_argument('--gamma', type=int, default=10,
                        help="Hyperparameter for length penalty.")
    parser.add_argument('--knowledge_source', type=str, default=None,
                        help="Name of the knowledge source for retrieval.")
    parser.add_argument('--cost_estimate', type=str, default="consider_cache",
                        choices=["consider_cache", "ignore_cache"],
                        help="Option to consider or ignore cache in cost estimation.")
    parser.add_argument('--abstain_detection_type', type=str, default=None,
                        choices=["perplexity_ai", "generic", "none"],
                        help="Type of abstain detection to use.")

    # Optional settings
    parser.add_argument('--use_atomic_facts', action='store_true',
                        help="Flag to use pre-existing atomic facts in the input data.")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose mode with progress bars.")
    parser.add_argument('--print_rate_limit_error', action='store_true',
                        help="Print rate limit errors when using OpenAI keys.")
    parser.add_argument('--n_samples', type=int, default=None,
                        help="Limit the number of samples to process.")
    parser.add_argument('--debug_logger', action='store_true',
                        help="Set logger level to debug")

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=os.path.join(os.getcwd(), __file__.replace(".py",".log")),
                        level=logging.DEBUG if args.debug_logger else logging.CRITICAL)

    logger = logging.getLogger(__name__)
    # Initialize FactScorer with parsed arguments
    fs = FactScorer(
        afv_model=args.afv_model,
        afg_model=args.afg_model,
        is_npm=args.is_npm,
        is_retrieval=args.is_retrieval,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        cache_dir=args.cache_dir,
        openai_key=args.openai_key,
        cost_estimate=args.cost_estimate,
        abstain_detection_type=args.abstain_detection_type
    )

    topics, generations, atomic_facts = [], [], []
    tot = 0
    logger.critical("Initialized FactScore")
    # Read input file
    with open(args.input_path, 'r', encoding='utf8') as f:
        for line in f:
            dp = json.loads(line)
            tot += 1
            if args.use_atomic_facts:
                assert "annotations" in dp, "`--use_atomic_facts` requires `annotations` in the input data."
                if dp["annotations"] is None:
                    continue
                topics.append(dp["topic"])
                generations.append(dp["output"])
                atomic_facts.append([
                    atom["text"] for sent in dp["annotations"] for atom in sent["llama-atomic-facts"]
                ])
            else:
                topics.append(dp["topic"])
                generations.append(dp["output"])
            if args.n_samples is not None and tot == args.n_samples:
                break

    logger.debug("Preparing to get scores")
    # Compute scores
    results = fs.get_score(
        topics=topics,
        generations=generations,
        gamma=args.gamma,
        atomic_facts=atomic_facts if args.use_atomic_facts else None,
        knowledge_source=args.knowledge_source,
        verbose=args.verbose
    )

    # Log results
    logging.critical("FactScore = %.1f%%", (100 * results["score"]))
    if "init_score" in results:
        logging.critical("FactScore w/o length penalty = %.1f%%", (100 * results["init_score"]))
    logging.critical("Respond ratio = %.1f%%", (100 * results["respond_ratio"]))
    logging.critical("# Atomic facts per valid response = %.1f", results["num_facts_per_response"])

    # Save results to output file
    output_path = args.input_path.replace(".jsonl", "_factscore_output.json")
    with open(output_path, 'w', encoding='utf8') as f:
        f.write(json.dumps(results, default=convert_to_serializable, indent=4) + "\n")

    print(f"Results saved to {output_path}")
