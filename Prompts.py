import os
path='/mnt/data/'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os
device = "cuda"
print("start_model")

model = AutoModelForCausalLM.from_pretrained(
    "Open-Orca/Mistral-7B-OpenOrca").to(device)
tokenizer = AutoTokenizer.from_pretrained(
    "Open-Orca/Mistral-7B-OpenOrca")

Document='''
Mercedes-Benz (German pronunciation: [mɛʁˌtseːdəs ˈbɛnts, -dɛs -] ⓘ),[6][7] commonly referred to as Mercedes and sometimes as Benz, is a German luxury and commercial vehicle automotive brand established in 1926. Mercedes-Benz AG (a Mercedes-Benz Group subsidiary established in 2019) is headquartered in Stuttgart, Baden-Württemberg, Germany.[1] Mercedes-Benz AG produces consumer luxury vehicles and light commercial vehicles badged as Mercedes-Benz. From November 2019 onwards, Mercedes-Benz-badged heavy commercial vehicles (trucks and buses) are managed by Daimler Truck, a former part of the Mercedes-Benz Group turned into an independent company in late 2021. In 2018, Mercedes-Benz was the largest brand of premium vehicles in the world, having sold 2.31 million passenger cars.[8]

The brand's origins lie in Daimler-Motoren-Gesellschaft's 1901 Mercedes and Carl Benz's 1886 Benz Patent-Motorwagen, which is widely regarded as the first internal combustion engine in a self-propelled automobile. The slogan for the brand is "the Best or Nothing".[9]

History
See also: List of companies involved in the Holocaust and Diesel emissions scandal

Karl Benz (1844–1929) made the 1886 Benz Patent Motorwagen, which is widely regarded as the first automobile.
Mercedes-Benz traces its origins to Karl Benz's first internal combustion engine in a car, seen in the Benz Patent Motorwagen – financed by Bertha Benz's dowry[10] and patented in January 1886[11] – and Gottlieb Daimler and their engineer Wilhelm Maybach's conversion of a stagecoach, with the addition of a petrol engine, introduced later that year. The Mercedes automobile was first marketed in 1901 by Daimler Motoren Gesellschaft (DMG).

Emil Jellinek-Mercedes, a Jewish-Austrian automobile entrepreneur who worked with DMG, registered the trademark in 1902, naming the 1901 Mercedes 35 hp after his daughter Mercedes Jellinek. Jellinek was a businessman and marketing strategist who promoted "horseless" Daimler automobiles among the highest circles of society. At the time, it was a meeting place for the haute volée of France and Europe, especially in winter. His customers included the Rothschild family and other wealthy clients, but as early as 1901, he was selling Mercedes cars in the "New World" as well, including to billionaires Rockefeller, Astor, Morgan, and Taylor. At the Nice race he attended in 1899, Jellinek drove under the pseudonym "Monsieur Mercédès". Many consider that race the birth of Mercedes-Benz as a brand. In 1901, the name "Mercedes" was re-registered by DMG worldwide as a protected trademark. The first Mercedes-Benz branded vehicles were produced in 1926, following the merger of Karl Benz and Gottlieb Daimler's companies into the Daimler-Benz company on 28 June of the same year.[11][12]


Gottlieb Daimler (1834–1900) – founder of Daimler-Motoren-Gesellschaft
Gottlieb Daimler was born on 17 March 1834 in Schorndorf. After training as a gunsmith and working in France, he attended the Polytechnic School in Stuttgart from 1857 to 1859. After completing various technical activities in France and England, he started working as a draftsman in Geislingen in 1862. At the end of 1863, he was appointed workshop inspector at a machine-tool factory in Reutlingen, where he met Wilhelm Maybach in 1865.[13]

Throughout the 1930s, Mercedes-Benz produced the 770 model, a car that was notably popular throughout Germany's Nazi period. Adolf Hitler was known to have driven in a model of this car during his time in power, with modified custom bulletproof windshields.[14] Most of the currently surviving 770 models were sold at auctions to private buyers. One of the cars is currently on display at the War Museum in Ottawa, Ontario.[15]

From 1937 onward, Daimler Benz focused increasingly on military products such as the LG3000 lorry and the DB600 and the DB601 aero engines. To build the latter, in 1936, it built a factory hidden in the forest at Genshagen around 10 km south of Berlin. By 1942, the company had mostly stopped producing cars, and was now devoted to war production. According to its statement, in 1944, almost half of its 63,610 employees were forced labourers, prisoners of war, or concentration-camp detainees.[16] Another source quotes this figure at 46,000. The company later paid $12 million in reparations to the labourers' families.[17]

In 1958, the two companies began a partnership to sell their cars in the United States with Studebaker. A few American-based Daimler Benz dealerships were converted into Mercedes-Benz dealerships when Daimler's non-Mercedes-partnered company closed in 1966.

Over the decades, Mercedes-Benz has introduced many electronic and mechanical innovations and safety features that later became common.[18] Currently, Mercedes-Benz is one of the best-known and longest-standing automotive brands in the world. The pontiff's Popemobile has often been sourced from Mercedes-Benz.[19]

In November 2019, Daimler AG announced that Mercedes-Benz, until that point a company marque, would be spun off into a separate, wholly owned subsidiary called Mercedes-Benz AG. The new subsidiary would manage the Mercedes-Benz car and van business. Mercedes-Benz-badged trucks and buses would be part of the Daimler Truck AG subsidiary.[1]

For information relating to the three-pointed star symbol of the brand, see under the title Daimler-Motoren-Gesellschaft, including the merger into Daimler-Benz.

In May 2022, Mercedes-Benz announced that it has recently sold the most expensive car at the price of $142 million (€135 million).[20] The car is a very rare 1955 Mercedes-Benz SLR that has been kept in the German automaker's collection and bought by a private owner. Mercedes in an announcement said that the sale will be used to establish the Mercedes-Benz Fund.[21]

In June 2022, Mercedes-Benz recalled almost one million vehicles built between 2004 and 2015, due to potential problems with their braking system, caused by possible "advanced corrosion".[22]

Mercedes-Benz ist eine eingetragene Handelsmarke für Automobile der Mercedes-Benz Group. Der Name entstand 1926 nach dem Zusammenschluss der Daimler-Motoren-Gesellschaft (Marke Mercedes) mit Benz & Cie. zur Daimler-Benz AG.[1] 2022 wurden 2,04 Millionen PKW-Neufahrzeuge der Marke verkauft

Die Bayerische Motoren Werke Aktiengesellschaft (BMW AG) ist ein börsennotierter Automobil- und Motorradhersteller mit Sitz in München, der auch als BMW Group auftritt. Die Produktpalette umfasst die Automobil- und Motorrad-Marke BMW, die Automarken Mini und Rolls-Royce sowie die BMW-Submarken BMW M und BMW i. BMW gehört mit 111 Milliarden Euro Umsatz und rund 118.900 Beschäftigten im Geschäftsjahr 2021[3] zu den größten Wirtschaftsunternehmen Deutschlands und zählt mit einer Jahresproduktion von 2,52 Millionen Automobilen und 118.900 Motorrädern im Jahr 2021[3] zu den 15 größten Kraftfahrzeugherstellern der Welt. Größte Anteilseigner mit zusammen etwa 48,5 % sind Susanne Klatten und Stefan Quandt, die der Industriellenfamilie Quandt angehören.[5]
Beginn
Vorgänger von BMW waren die 1913 von Karl Rapp gegründeten Rapp Motorenwerke GmbH. Als „Grundlage seines Unternehmens“ kaufte Rapp die Firma des aus Berlin stammenden Philipp Dörhöfer, die sich in der Münchener Clemensstraße 46 befand. „Diese wieder herum hatte kurz zuvor in Chemnitz die Firma Schneeweis erworben. Schneeweis baute Flugmotoren unter anderem für den Luftschiffbauer Albert Paul Veeh. Im Zuge der Insolvenz von Veeh war Schneeweis in Schwierigkeiten geraten. Diese waren offensichtlich so groß, dass sich auch die Firma Dörhöfer an Schneeweis verhob und Rapp das Unternehmen übernahm, um Flugmotoren zu fertigen.“[6] Grund für Dörhöfers Übernahme von Schneeweis’ Firma war, dass sie schon Flugmotoren für die Luftschiffbau Veeh GmbH baute und er damit auf jahrelange Erfahrungen zurückgreifen konnte. Später schrieb er an seinen Sohn: „Der Ursprung der BMW ist die Firma Schneeweis in Chemnitz, die über den Luftschiffbau Veeh und das Flugwerk Deutschland, wo auch mein Name genannt ist, die Rapp-Motorenwerke hervorbrachten.“[7] Die daraus entstandene Rapp Motorenwerke GmbH änderte ihren Namen im April 1917 zunächst in BMW GmbH und ein Jahr später, nach der Umwandlung in eine Aktiengesellschaft, in BMW AG. Die bisherige GmbH ging in Liquidation. Die Errichtung der AG war mit der Heeresverwaltung abgesprochen.[8] Der erste Geschäftsführer war bis 1942 Franz Josef Popp.

Im jungen Unternehmen machte sich der aufstrebende Ingenieur Max Friz schnell einen Namen: Er entwickelte 1917 den Flugmotor BMW IIIa mit Überverdichtung, die den Leistungsverlust in der Höhe verringert. Diese Konstruktion bewährte sich insbesondere im Jagdflugzeug Fokker D.VII so gut, dass BMW von der Heeresverwaltung einen Auftrag über 2000 Motoren erhielt. Am 17. Juni 1919 wurde mit einem BMW IV, einer Weiterentwicklung des BMW IIIa ein inoffizieller Höhenweltrekord (Deutschland war nicht Mitglied der FAI) von 9760 Metern erzielt.[9]

Mit dem Ende des Ersten Weltkrieges und dem Versailler Vertrag schien zunächst das Ende des Unternehmens gekommen zu sein: der Friedensvertrag verbot es für fünf Jahre, in Deutschland Flugmotoren – damals das einzige Produkt von BMW – herzustellen. Werbeinserate von 1920 zeigen jedoch, dass BMW nicht ganz dem Verbot folgte.[10][11]

1922 verließ Hauptaktionär Camillo Castiglioni das Unternehmen und nahm die Namensrechte an BMW mit. Er ging zu den Bayerischen Flugzeugwerken (BFW). Diese waren aus den am 7. März 1916 registrierten Bayerischen Flugzeugwerken hervorgegangen, die sich wiederum aus dem Anfang des Jahres in Konkurs gegangenen Gustav-Otto-Flugzeugwerk von Gustav Otto, einem Sohn des Ottomotor-Erfinders Nikolaus Otto, entwickelt hatten. Dieser 7. März 1916 gilt in der offiziellen Unternehmensgeschichtsschreibung als Gründungsdatum von BMW. Mit dem Wechsel von Castiglioni werden aus den Bayerischen Flugzeugwerken (BFW) BMW, am Firmensitz Lerchenauer Straße 76, München 13.[12] Das Unternehmen aber, das bis dahin BMW hieß, wurde zur Südbremse und später dann zur Knorr-Bremse.

1923 entwickelten Max Friz und Martin Stolle das erste BMW-Motorrad, die R 32, und legten damit den Grundstein für eine neue Produktionslinie: Motorräder. Friz brauchte für den Entwurf der R 32 nur fünf Wochen. Bis heute hat sich das Grundprinzip dieses Motorrades erhalten: Boxermotor und Kardanantrieb im Doppelrohrrahmen.

Ab 1924 wurden auch wieder Flugmotoren hergestellt. Der 1930 im Reichsbahn-Ausbesserungswerk Hannover-Leinhausen gebaute „Schienenzeppelin“ wurde von einem BMW-VI-Motor angetrieben.

Start als Automobilhersteller in Eisenach
Im Jahr 1928 übernahm BMW die Fahrzeugfabrik Eisenach A. G., den Hersteller des Kleinwagens Dixi, und wurde so zum Automobil­hersteller. Am 22. März 1929 produzierte BMW im thüringischen Eisenach sein erstes Serienautomobil. Das Modell hieß 3/15 PS bzw. DA 2 und war eine Weiterentwicklung des Modells Dixi 3/15 DA, das seinerseits ein modifizierter Lizenzbau des britischen Austin Seven war. Der Wagen wurde in Berlin mit einer von Ambi-Budd gelieferten Karosserie, die dem ebenfalls in Austin-Lizenz gebauten Rosengart ähnelte, montiert. 1932 folgte der erste „echte“ BMW der BMW AM-Baureihe mit der Bezeichnung AM1 (für „Automobilkonstruktion München Nr. 1“), d. h. die erste BMW-eigene Automobilkonstruktion, die gegenüber dem BMW 3/15 größer und technisch fortschrittlicher ausfiel (z. B. obengesteuerte Ventile, Vierradbremse, Schwingachse vorn). Die erste Neukonstruktion unter der BMW-Ägide war der 1933 vorgestellte 303 mit 1,2 Liter Sechszylindermotor, eine Konstruktion von Fritz Fiedler (1899–1972). Infolge des ab 1933 wieder stark erweiterten Flugmotorenbaus wurde die Auto- und Motorradsparte fast zum Nebenzweck. Trotzdem gelangen mit den Neuentwicklungen BMW 326 (1935), 327 (1937) und dem 1936 vorgestellten Sport-Roadster 328 attraktive Modelle. Besonders der 328 überzeugte nicht nur durch seine Konstruktion, sondern auch durch zahlreiche Erfolge bei Sportwagenrennen, unter anderem der Mille Miglia 1940. Dieses Modell begründete den Ruf von BMW als Hersteller sportlicher Automobile, der auch nach dem Krieg in Erinnerung blieb. In Großbritannien wurde der 328 als Frazer-Nash-BMW vermarktet, wobei Frazer Nash bereits seit 1934 als BMW-Generalimporteur für das britische Empire fungierte. Die Baupläne der 326/327/328-Reihe dienten später der Entwicklung des Bristol 400.

Bis zum Ende des Zweiten Weltkriegs

BMW 801

Aktie von BMW, 1942
Nach der Machtergreifung der Nationalsozialisten erfuhr BMW einen kräftigen Aufschwung durch die Kriegspläne Hitlers. Während Mitte 1933 noch 8.357 Leute in der deutschen Flugzeug- und Flugmotorenindustrie ihren Arbeitsplatz hatten, war die Beschäftigtenzahl Ende 1938 auf fast 180.000 angewachsen. An diesem Aufschwung partizipierte auch BMW. Der Umsatz des Unternehmens betrug 32,5 Millionen Reichsmark (RM) im Jahr 1933 und steigerte sich bis 1939 auf 280 Millionen RM. Der Flugmotorenbau bei der 1934 neu gegründeten Tochtergesellschaft „BMW Flugmotorenbau GmbH“ erfolgte in der neuen BMW Flugmotorenfabrik Allach GmbH (heute MTU Aero Engines) und der BMW Flugmotorenfabrik Eisenach GmbH (Dürrerhof – nach Kriegsende demontiert). Dieser trug 1939 allein 190 Millionen RM zum Umsatz bei. Mit der Übernahme der Brandenburgischen Motorenwerke in Berlin-Spandau im Jahr 1939, die anschließend als BMW Flugmotorenwerke Brandenburg GmbH firmierten, und der Gründung der Niederbarnimer Flugmotorenwerke GmbH im Jahr 1941 mit Standorten in Zühlsdorf und Basdorf expandierte der Geschäftsbereich Flugmotoren auf 90 Prozent des gesamten Umsatzes. Im Jahr 1944 wurden 750 Millionen RM Umsatz von zirka 56.000 Beschäftigten, rund 50 Prozent davon waren Zwangsarbeiter, erwirtschaftet.

In den Werken München und Eisenach wurden „schwere Wehrmachtsgespanne“ (Motorräder BMW R 75 mit angetriebenem Beiwagen) und zwischen 1937 und 1940 der leichte geländegängige Einheits-PKW BMW 325 gebaut. Letzter musste nach Vorgaben der Wehrmacht in weitgehend identischer Konstruktion auch von Stoewer und Hanomag hergestellt werden.

Im Rahmen der Aufrüstung wurde ab 1936 ein neues Werk in Allach nahe München errichtet. Das Werk München-Allach wurde von Beginn an in Tarnbauweise gebaut, war als reines Flugmotorenwerk konzipiert[13] und war bis 1938 vor allem als Ergänzung zum Werk München gedacht.[14] Ab 1940/41 wurde das Werk massiv erweitert und die Serienfertigung von Flugmotoren begonnen.[15][16] BMW setzte dort zum Ausbau des Werkes und zur Fertigung von Flugmotoren Zwangsarbeiter und ab 1942 auch KZ-Häftlinge ein.[17][18] Untergebracht waren diese in Zwangsarbeitslagern und im KZ-Außenlager München-Allach des KZ Dachau. 1944 waren im Werk Allach 17.313 Menschen beschäftigt, davon waren 11.623 (67,1 %) Zwangsarbeiter.[14]

Der bis zu 1.467 kW (2.000 PS) starke Doppel-Sternmotor BMW 801 war einer der wichtigsten deutschen Flugmotoren. Er wurde unter anderem in die Focke-Wulf Fw 190 und Junkers Ju 88 eingebaut. Zeitweise waren in seiner Produktion zur Hälfte russische Zwangsarbeiter eingesetzt. Stückzahl und Leistung der BMW-801-Motoren mussten gesteigert werden. Erst im Jahr 1943 konnte das Unternehmen das gewünschte Produktionssoll erfüllen. Die Luftwaffe beklagte indessen unter anderem Kolbenfresser, Ventilschäden oder Kipphebelbrüche bei diesem Motor. Weitere Flugmotoren waren der BMW 132, BMW 802 und BMW 803.

Die Situation des Unternehmens war durch den Krieg aber auch beeinträchtigt. Allein im Frühjahr 1943 wurden 6.189 Beschäftigte zur Wehrmacht eingezogen, was den Verlust wichtigen Fachwissens in der Produktion bedeutete. Ausgeglichen wurde dies durch zunehmenden Einsatz von KZ-Häftlingen. Luftangriffe der Alliierten auf die kriegswichtigen Werke in Milbertshofen (Stammwerk) und Allach (neues Flugmotorenwerk) störten die Motorenherstellung empfindlich. Das Reichsluftfahrtministerium verfügte, in dem sieben Kilometer langen Tunnel der Eisenbahnstrecke Sélestat – Saint-Dié bei Markirch im Elsass die Fertigung fortzusetzen. 1.016 Maschinen wurden dorthin transportiert und 3.000 Menschen in neuer Umgebung eingesetzt. Mit dem Näherrücken der Alliierten wurde dieses Projekt wieder beendet und die Herstellung nach Süddeutschland verlagert, mit angeschlossenen KZ-Außenlagern. In Kempten (Allgäu) wurde das Werk des Zulieferers Helmuth Sachse KG[19] zur Zahnradfertigung bestimmt, in Blaichach wurden Pleuelteile für Flugmotoren erzeugt, weitere Produktionsstätten in Kaufbeuren, Immenstadt und den oberbayerischen Orten Trostberg und Stephanskirchen eingerichtet.[20]

Nachkriegszeit

R 68 (1954) mit Steib-Seitenwagen (1951)
1945 war das Münchner Stammwerk fast völlig zerstört und die Fahrzeugfabrik in Eisenach von der Sowjetischen Besatzungsmacht übernommen worden. Da das Eisenacher Automobilwerk im Besitz aller Produktionswerkzeuge war, konnte es sofort nach dem Krieg die Vorkriegs-Typen wieder anbieten. Dies geschah zunächst auch unter dem Namen „BMW“. Da BMW in München es nicht hinnehmen wollte, dass unter diesem Namen Autos angeboten wurden, ohne auf deren Produktion Einfluss zu haben, ließ man den Eisenachern 1951 das Führen des Namens „BMW“ gerichtlich verbieten. Die Eisenacher Fabrikate wurden daraufhin unter dem Namen „EMW“ (Eisenacher-Motoren-Werk) angeboten. 1952 wurde das Werk zum Volkseigenen Betrieb (VEB) erklärt.[21] Aufgrund zentraler Planvorgaben wurde das Eisenacher Werk gezwungen, die Produktion größerer Viertaktwagen zugunsten kleinerer Zweitakt-Fahrzeuge auf Basis des IFA F 9 umzustellen. 1955 rollten die letzten EMW 340 vom Band. Fortan produzierte der nunmehr VEB Automobilwerk Eisenach genannte Betrieb den Wartburg.

In München waren bis dato nie Automobile produziert worden, zusätzlich war das Stammwerk zerbombt und von Demontagen betroffen. Zunächst hielt sich das Unternehmen mit der Fabrikation von Motorrädern, Kochtöpfen und Fahrzeugbremsen über Wasser. 1948 brachte BMW mit der R 24 sein erstes Motorrad nach dem Krieg auf den Markt, 1952 gefolgt vom BMW 501, einem exklusiven Oberklassewagen mit Sechszylindermotor. Der ab 1954 auch mit V8-Motor als BMW 502 erhältliche Pkw erhielt wegen seiner geschwungenen Karosserieform bald den Spitznamen „Barockengel“. Die Produktion des Typs war so aufwendig, dass BMW bei jedem verkauften Exemplar zirka 4.000 DM Verlust einfuhr. Ein weiteres Problem war der ab Mitte der 1950er Jahre stark rückläufige Motorrad-Absatz. Auch der 1955 in Produktion genommene Kleinstwagen Isetta, eine Lizenzproduktion des italienischen Herstellers Iso Rivolta, konnte die sich schnell verschärfende Finanzkrise nicht abwenden. 1957 wurde das alte Werk 2 in München-Allach an MAN verkauft.[22]

Krise und Beinaheübernahme
Nachdem in den Geschäftsjahren 1958 und 1959 hohe Verluste erwirtschaftet worden waren, kam es zu der dramatischen Hauptversammlung vom 9. Dezember 1959. Vorstand und Aufsichtsrat, beide von der Deutschen Bank eingesetzt, legten ein Angebot vor, nach dem BMW an die Daimler-Benz AG (Großaktionär ebenfalls Deutsche Bank) verkauft und die Kleinaktionäre fast enteignet worden wären. Das Schicksal von BMW schien besiegelt, da die Deutsche Bank dank des Depotstimmrechts etwa die Hälfte des Aktienkapitals vertrat. Aber es kam anders: Eine Ablehnungsfront, gebildet aus Belegschaft und Betriebsräten, BMW-Händlern und Kleinaktionären, wehrten das Übernahme-Angebot ab, indem sie mit Hilfe des Darmstädter Aktionärs und Kohlenhändlers Erich Nold (1928–1995[23]) sowie des Frankfurter Rechtsanwalts Friedrich Mathern die Bilanz anfechten ließen, wofür 10 Prozent der Stimmen genügten. Die Bilanz war in der Tat fehlerhaft, da in ihr die Entwicklungskosten für das neue Modell 700 innerhalb eines Jahres abgeschrieben worden waren.[24] Für ein Sanierungs- und Investitionsprogramm benötigte BMW dringend Kapital, welches 1958 durch die Ausgabe von Schuldverschreibungen im Wert von 15 Millionen Mark eingeworben wurde[25][26]. Nachdem sich für die Papiere zunächst keine Käufer fanden, kaufte der Bremer Unternehmer Hermann Krages, der bereits 25 Prozent an BMW besaß[27], sämtliche Schuldverschreibungen auf[28].

'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer

import gc
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from langchain.text_splitter import CharacterTextSplitter

# Document
count=0

print("started")
try:
    del outputs
except:
    pass
try:
    del text
except:
    pass
a=gc.collect()
torch.cuda.empty_cache()

write=open(path+"/try/"+"test.txt","w",encoding='utf-8')
data= Document
message=""
header=""
linenumber=0
for line in data:
    message=message+line

max_tokens = 400
# Optionally can also have the splitter not trim whitespace for you
tokenizer22 = Tokenizer.from_pretrained("bert-base-uncased")
splitter = HuggingFaceTextSplitter(tokenizer22, trim_chunks=True)
chunks = splitter.chunks(message, max_tokens)
texts = chunks

gc.collect()
torch.cuda.empty_cache()            
del chunks
gc.collect()
torch.cuda.empty_cache()
QA=""
for each in texts:
    each= header+ " \n \n "+each
    each=each.replace("   "," ")
    each=each.lstrip().rstrip().strip()

    prompt = "I give you a text. If it is not in English, translate it into English. You must write twenty 'Q_A' for me. Each 'Q_A' must have a question with an answer to that question. You must always mention the entire name of the company for each question. The text for generating Q_A is as follows: \n\n"+each+" \n ****++++**** \n"#.replace("\n","")
    prompt=prompt.replace("  "," ")
    prompt=prompt.lstrip().rstrip().strip()
    inputs = tokenizer22(
        prompt,
        return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs, max_new_tokens=2300, use_cache=True, do_sample=True,
        temperature=0.2, top_p=0.95)

    bbb=tokenizer22.batch_decode(outputs)[0].replace(each,"").lstrip().rstrip().strip().split("****++++****")[-1]

    QA=QA+ " \n \n "+bbb+" \n \n "
    del outputs
    del inputs
    gc.collect()
    torch.cuda.empty_cache()

text =message+ " \n $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n \n "+ QA +"\n"
del QA
gc.collect()
torch.cuda.empty_cache()
print("+++++++++++++++++++++ done +++++++++++++++++++++++++")
write.write(text)
write.flush()
write.close()
count=count+1
try:
    del text
except:
    pass
try:
    del outputs
except:
    pass
gc.collect()
torch.cuda.empty_cache()
gc.collect()
del texts
torch.cuda.empty_cache()
