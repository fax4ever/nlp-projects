import pytest
from sentence_splitter_llm.minerva_labels import MinervaLabels
from datasets import load_dataset


@pytest.fixture
def minerva_output():
    return "<s><s><|start_header_id|> system<|end_header_id|> \n\nSei un esperto di linguistica italiana specializzato nella segmentazione delle frasi.<|eot_id|><|start_header_id|> user<|end_header_id|> \n\nDividi il seguente testo italiano in frasi. Per favore rispondi con una frase per riga. Grazie.\n\nTesto: C'era una volta ... – Un re! – diranno subito i miei piccoli lettori. – No, ragazzi, avete sbagliato. C'era una volta un pezzo di legno. Non era un legno di lusso, ma un semplice pezzo da catasta, di quelli che d'inverno si mettono nelle stufe e nei caminetti per accendere il fuoco e per riscaldare le stanze. Non so come andasse, ma il fatto gli è che un bel giorno questo pezzo di legno capitò nella bottega di un vecchio falegname, il quale aveva nome mastr'Antonio, se non che tutti lo chiamavano maestro Ciliegia, per via della punta del suo naso, che era sempre lustra e paonazza, come una<|eot_id|><|start_header_id|> assistant<|end_header_id|> \n\n1. C'era una volta ... –\n2. Un re! – diranno subito i miei piccoli lettori.\n3. – No, ragazzi, avete sbagliato.\n4. C'era una volta un pezzo di legno.\n5. Non era un legno di lusso, ma un semplice pezzo da catasta, di quelli che d'inverno si mettono nelle stufe e nei caminetti per accendere il fuoco e per riscaldare le stanze.\n6. Non so come andasse, ma il fatto gli è che un bel giorno questo pezzo di legno capitò nella bottega di un vecchio falegname, il quale aveva nome mastr'Antonio, se non che tutti lo chiamavano maestro Ciliegia, per via della punta del suo naso, che era sempre lustra e paonazza, come una<|eot_id|>"
    

@pytest.fixture
def words_and_labels():
    dataset_dict = load_dataset("fax4ever/sentence-splitter-ood-128")
    return dataset_dict["test"][0]


@pytest.fixture
def words(words_and_labels):
    return words_and_labels["tokens"]


@pytest.fixture
def golden_labels(words_and_labels):
    return words_and_labels["labels"]    

    
def test_minerva_labels(minerva_output, words):
    minerva_labels = MinervaLabels(minerva_output, words)
    sentences = minerva_labels.sentences
    assert len(sentences) == 6
    assert sentences[0] == "C'era una volta ... –"
    assert sentences[1] == "Un re! – diranno subito i miei piccoli lettori."
    assert sentences[2] == "– No, ragazzi, avete sbagliato."
    assert sentences[3] == "C'era una volta un pezzo di legno."
    assert sentences[4] == "Non era un legno di lusso, ma un semplice pezzo da catasta, di quelli che d'inverno si mettono nelle stufe e nei caminetti per accendere il fuoco e per riscaldare le stanze."
    assert sentences[5] == "Non so come andasse, ma il fatto gli è che un bel giorno questo pezzo di legno capitò nella bottega di un vecchio falegname, il quale aveva nome mastr'Antonio, se non che tutti lo chiamavano maestro Ciliegia, per via della punta del suo naso, che era sempre lustra e paonazza, come una"


def test_minerva_aligned_sentences(minerva_output, words):
    minerva_labels = MinervaLabels(minerva_output, words)
    aligned_sentences = minerva_labels.aligned_sentences
    assert len(aligned_sentences) == 6
    assert aligned_sentences[0] == "C'eraunavolta…–"
    assert aligned_sentences[1] == "Unre!–dirannosubitoimieipiccolilettori."
    assert aligned_sentences[2] == "–No,ragazzi,avetesbagliato."
    assert aligned_sentences[3] == "C'eraunavoltaunpezzodilegno."
    assert aligned_sentences[4] == "Noneraunlegnodilusso,maunsemplicepezzodacatasta,diquelliched'invernosimettononellestufeeneicaminettiperaccendereilfuocoeperriscaldarelestanze."
    assert aligned_sentences[5] == "Nonsocomeandasse,mailfattogliècheunbelgiornoquestopezzodilegnocapitònellabottegadiunvecchiofalegname,ilqualeavevanomemastr'Antonio,senonchetuttilochiamavanomaestroCiliegia,perviadellapuntadelsuonaso,cheerasemprelustraepaonazza,comeuna"


def test_minerva_aligned_labels(minerva_output, words, golden_labels):
    minerva_labels = MinervaLabels(minerva_output, words)
    aligned_labels = minerva_labels.aligned_labels(golden_labels)

    print(aligned_labels)
    print(golden_labels)
    assert aligned_labels is not None
