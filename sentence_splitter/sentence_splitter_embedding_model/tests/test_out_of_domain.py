from sentence_splitter_embedding_model.evalutation import eval_model
from pathlib import Path


def test_eval_model():
    value = eval_model("bert-base-cased-sentence-splitter", Path(__file__).parent / "Cuore-GOLD.txt", 6)
    print(value)
    print(value["f1"])


def test_text():
    text = """Non era un legno di lusso, ma un semplice pezzo
da catasta, di quelli che d’inverno si mettono nelle
stufe e nei caminetti per accendere il fuoco e per riscaldare le stanze.
Non so come andasse, ma il fatto gli è che un bel
giorno questo pezzo di legno capitò nella bottega
di un vecchio falegname, il quale aveva nome mastr’Antonio, se non che tutti lo chiamavano maestro
Ciliegia, per via della punta del suo naso, che era
sempre lustra e paonazza, come una ciliegia matura.
Appena maestro Ciliegia ebbe visto quel pezzo di
legno, si rallegrò tutto; e dandosi una fregatina di
mani per la contentezza, borbottò a mezza voce:
"Questo legno è capitato a tempo; voglio servirmene per fare una gamba di tavolino." 
"""
    text = text.splitlines()
    text = " ".join(text)
    print(text)