import json
import subprocess
import sys
from pathlib import Path

from data.link_context import (CONTEXT_WORDS, add_context, extract, is_navigation, is_prose,
                               paragraphs, remove_captions, title_pattern)

TEXT = """   #copyright

18th century

2007 Schools Wikipedia Selection. Related subjects: General history

   Centuries:         17th century - 18th century - 19th century and more navigation entries here

   As a means of recording the passage of time, the 18th century refers to
   the century that lasted from 1701 through 1800 in the Gregorian
   calendar. Historians link it to the French Revolution.

   The period runs between the Glorious Revolution in England and the
   start of the French Revolution.
   Enlarge

     * 1701- 1714: War of the Spanish Succession
     * 1703: Saint Petersburg founded by Peter the Great. Russian capital
       until 1918.
     * Mercury
"""
LINKS = {"17th_century": "17th century", "French_Revolution": "French Revolution",
         "Glorious_Revolution": "Glorious Revolution", "England": "England",
         "War_of_the_Spanish_Succession": "War of the Spanish Succession",
         "Saint_Petersburg": "Saint Petersburg", "Time": "Time", "Mercury_(planet)": "Mercury (planet)",
         "Steam_engine": "Steam engine"}


def test_boilerplate_is_dropped_and_list_entries_stand_alone():
    blocks = paragraphs(TEXT, "18th century")
    assert not any("copyright" in b or "Schools Wikipedia" in b or b == "18th century" for b in blocks)
    assert "1701- 1714: War of the Spanish Succession" in blocks
    assert "1703: Saint Petersburg founded by Peter the Great. Russian capital until 1918." in blocks


def test_lead_is_the_first_prose_paragraph():
    lead = extract(TEXT, "18th century", LINKS)["lead"]
    assert lead.startswith("As a means of recording the passage of time")  # Not the navigation line.


def test_links_in_first_mention_order_with_their_sentence():
    result = extract(TEXT, "18th century", LINKS)
    assert result["link_order"] == ["17th_century", "Time", "French_Revolution", "Glorious_Revolution",
                                    "England", "War_of_the_Spanish_Succession", "Saint_Petersburg",
                                    "Mercury_(planet)"]
    context = result["link_context"]
    assert context["French_Revolution"] == "Historians link it to the French Revolution."
    assert context["Saint_Petersburg"] == "1703: Saint Petersburg founded by Peter the Great."
    # Lowercase mentions count: Wikipedia links common nouns like "time" in lowercase.
    assert context["Time"].startswith("As a means of recording the passage of time")
    assert "Steam_engine" not in result["link_order"]  # Never mentioned: title only in the prompt.
    assert "Mercury_(planet)" not in context            # A bare one-word entry carries no sentence.


def test_title_matching_rules():
    assert title_pattern("Mercury (planet)").search("Mercury orbits the Sun")
    assert title_pattern("Time").search("at that time")
    assert not title_pattern("Time", ignore_case=False).search("at that time")
    assert title_pattern("French Revolution").search("the french revolution began")
    assert not title_pattern("Spain").search("Spaniards")


def test_long_sentences_are_cut_around_the_mention():
    sentence = " ".join(f"w{i}" for i in range(40)) + " Target " + " ".join(f"z{i}" for i in range(40)) + "."
    context = extract(sentence, "Page", {"Target": "Target"})["link_context"]["Target"]
    assert context.startswith("...") and context.endswith("...") and "Target" in context.split()
    assert len(context.replace("...", " ").split()) == CONTEXT_WORDS


def test_add_context_keeps_fields_and_reports_coverage():
    articles = {"Century": {"title": "18th century", "text": TEXT, "outgoing": list(LINKS)}}
    articles.update({link: {"title": title, "text": "", "outgoing": []} for link, title in LINKS.items()})
    result, stats = add_context(articles)
    assert result["Century"]["text"] == TEXT and result["England"]["link_order"] == []
    assert stats["outgoing_links"] == len(LINKS) and stats["links_mentioned_in_text"] == 8
    assert stats["links_with_context"] == 6  # 17th century appears only in a navigation line.


def test_cli_writes_a_new_file_only(tmp_path):
    root = Path(__file__).resolve().parents[1]
    articles = tmp_path / "articles.json"
    data = {"Century": {"title": "18th century", "text": TEXT, "outgoing": ["England"]},
            "England": {"title": "England", "text": "", "outgoing": []}}
    articles.write_text(json.dumps(data), encoding="utf-8")
    run = lambda output: subprocess.run(
        [sys.executable, "-m", "data.link_context", "--articles", str(articles), "--output", str(output)],
        cwd=root, capture_output=True, text=True)
    assert run(articles).returncode != 0  # Never overwrites articles.json.
    output = tmp_path / "articles_context.json"
    completed = run(output)
    assert completed.returncode == 0, completed.stderr
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["Century"]["link_order"] == ["England"]



MANET = """Édouard Manet

   Articles with similar titles include Claude Monet, another painter of the same era.
   Édouard Manet portrait by Nadar Birth name Édouard Manet Born January 23, 1832 Paris Died April 30,
   1883 Paris Nationality French Field Painting Famous works The Luncheon on the Grass

   Satellite view of the city of Paris and its suburbs.
   Enlarge
   Satellite view of the city of Paris and its suburbs.
   Édouard Manet ( January 23, 1832 – April 30, 1883) was a French painter. One of the first
   nineteenth century artists to approach modern-life subjects, he was a pivotal figure in the
   transition from Realism to Impressionism.

   Manet was born in Paris to an upper-class household. He studied painting under Thomas Couture.
"""


def test_captions_are_removed_but_their_paragraph_kept():
    text = "Intro sentence here. A map of the region (shaded). Enlarge A map of the region (shaded). The king ruled it."
    assert remove_captions(text) == "Intro sentence here. The king ruled it."


def test_infobox_rows_are_not_prose():
    assert not is_prose("Birth name Édouard Manet Born January 23, 1832 Paris Died April 30, 1883 Paris")
    assert is_prose("He visited Germany, Italy, and the Netherlands, absorbing the influence of Frans Hals.")


def test_prose_is_preferred_over_infoboxes_hatnotes_and_captions():
    links = {"Paris": "Paris", "Painting": "Painting", "Claude_Monet": "Claude Monet",
             "Impressionism": "Impressionism"}
    result = extract(MANET, "Édouard Manet", links)
    assert result["lead"].startswith("Édouard Manet ( January 23, 1832 – April 30, 1883) was a French painter.")
    context = result["link_context"]
    assert context["Paris"] == "Manet was born in Paris to an upper-class household."   # Not the caption or infobox.
    assert context["Painting"] == "He studied painting under Thomas Couture."           # Lowercase prose mention.
    assert "Claude_Monet" not in result["link_order"]                                   # Only in the hatnote.


def test_infobox_row_is_a_fallback_when_prose_never_mentions_the_link():
    text = "Born January 23, 1832 Lyon Died April 30, 1883 Paris Nationality French Field Painting\n\nHe was a painter of renown."
    assert extract(text, "X", {"Lyon": "Lyon"})["link_context"]["Lyon"].startswith("Born January 23, 1832 Lyon")



C18 = """   #copyright

18th century

   Millennium: 1st millennium - 2nd millennium - 3rd millennium
   Centuries:         17th century - 18th century - 19th century

   As a means of recording the passage of time, the 18th century refers to
   the century that lasted from 1701 through 1800 in the Gregorian calendar.

Significant people

   Johann Sebastian Bach
   Enlarge
   Johann Sebastian Bach
     * Johann Sebastian Bach, German composer
     * Immanuel Kant, German philosopher
"""


def test_navigation_lines_never_become_a_links_sentence():
    assert is_navigation("Millennium: 1st millennium - 2nd millennium - 3rd millennium")
    assert not is_navigation("1779- 1879: Xhosa Wars between British and Boer settlers")
    result = extract(C18, "18th century", {"17th_century": "17th century", "Time": "Time"})
    assert result["link_order"][0] == "17th_century"          # Still in page order...
    assert "17th_century" not in result["link_context"]       # ...but with its title only.
    assert result["link_context"]["Time"].startswith("As a means of recording the passage of time")


def test_list_entries_are_prose_and_captions_disappear():
    assert is_prose("1703: Saint Petersburg founded by Peter the Great.")
    result = extract(C18, "18th century", {"Bach": "Johann Sebastian Bach", "Kant": "Immanuel Kant"})
    assert result["link_context"]["Bach"] == "Johann Sebastian Bach, German composer"  # Not the caption.
    assert result["link_context"]["Kant"] == "Immanuel Kant, German philosopher"       # Fallback entry.
