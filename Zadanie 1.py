import random
import time
import tkinter as tk

class Jedinec:
    def __init__(self):
        # Atribút adresy bude obsahovať 6-bitové binárne čísla od 0 do 63
        self.adresy = [format(i, '06b') for i in range(64)]
        # Atribút hodnoty bude obsahovať náhodné 8-bitové binárne čísla
        self.hodnoty = [format(random.randint(0, 255), '08b') for i in range(64)]
        # Atribút kroky bude obsahovať ako sa jedinec pohybuje, zo začiatku nastavené hodnoty na X
        self.kroky = ['X' for i in range(500)]
        self.pocet_krokov = 0
        self.fitness = 100
        self.pocet_pokladov = 0

def riesenie_jedinca(jedinec, poradie):
    # Uložíme si pôvodný stav hodnôt pred riešením
    povodne_hodnoty = jedinec.hodnoty.copy()

    # na akej hodnote sa nachádzame
    index = 0
    # na akom kroku sa nachádzame
    krok_index = 0
    instrukcie_limit = 500
    instrukcie_vykonane = 0

    while index < len(jedinec.hodnoty):
        # Ukonči cyklus, keď dosiahneme limit
        if instrukcie_vykonane >= instrukcie_limit:
            break

        hodnota = jedinec.hodnoty[index]

        if hodnota.startswith("00"):
            adresa_bin = hodnota[2:]  # Posledných 6 bitov
            adresa = int(adresa_bin, 2)  # Preveď na celé číslo
            aktualna_hodnota_bin = jedinec.hodnoty[adresa]
            nova_hodnota = format((int(aktualna_hodnota_bin, 2) + 1) % 256, '08b')
            jedinec.hodnoty[adresa] = nova_hodnota

        elif hodnota.startswith("01"):
            adresa_bin = hodnota[2:]
            adresa = int(adresa_bin, 2)
            aktualna_hodnota_bin = jedinec.hodnoty[adresa]
            nova_hodnota = format((int(aktualna_hodnota_bin, 2) - 1) % 256, '08b')
            jedinec.hodnoty[adresa] = nova_hodnota

        elif hodnota.startswith("10"):
            adresa_bin = hodnota[2:]
            adresa = int(adresa_bin, 2)
            index = adresa
            instrukcie_vykonane += 1
            continue  # Znova spracuje hodnotu na novej adrese

        elif hodnota.startswith("11"):
            jedinec.pocet_krokov += 1
            pocet_jednotiek = hodnota.count('1') # Zisti počet jednotiek na adrese

            # Krok hore
            if pocet_jednotiek <= 2:
                if krok_index < len(jedinec.kroky):
                    jedinec.kroky[krok_index] = 'H'
                krok_index += 1
            # Krok dole
            elif 3 <= pocet_jednotiek <= 4:
                if krok_index < len(jedinec.kroky):
                    jedinec.kroky[krok_index] = 'D'
                krok_index += 1
            # Krok doprava
            elif 5 <= pocet_jednotiek <= 6:
                if krok_index < len(jedinec.kroky):
                    jedinec.kroky[krok_index] = 'P'
                krok_index += 1
            # Krok dolava
            elif 7 <= pocet_jednotiek <= 8:
                if krok_index < len(jedinec.kroky):
                    jedinec.kroky[krok_index] = 'L'
                krok_index += 1

            index += 1
            instrukcie_vykonane += 1
            continue  # Pokračuje do ďalšej iterácie cyklu

        index += 1
        instrukcie_vykonane += 1  # Zvýši počet inštrukcií

    # Po riešení obnovíme pôvodné hodnoty
    jedinec.hodnoty = povodne_hodnoty

def prechadzka(jedinec, poradie, prvy_krat):
    # Vytvorenie 7x7 matice s nulami
    mriezka = [[0 for j in range(7)] for i in range(7)]

    # Pozície pokladov (indexované od 0)
    poklady = [(1, 4), (2, 2), (3, 6), (4, 1), (5, 4)]

    # Pridanie pokladov do matice
    for (x, y) in poklady:
        mriezka[x][y] = 1  # Označíme poklad číslom 1

    # Štartovacia pozícia jedinca
    x = 6
    y = 3

    for krok in jedinec.kroky:
        if krok == "H":
            jedinec.fitness -= 1
            if x > 0:
                x -= 1
            else:
                #print("Jedinec vysiel z mriezky H")
                jedinec.fitness -= 10
                x -= 1
                break
            if mriezka[x][y] == 1:
                #print("Jedinec nasiel poklad")
                mriezka[x][y] = 0
                jedinec.fitness += 30
                jedinec.pocet_pokladov += 1
        elif krok == "D":
            jedinec.fitness -= 1
            if x < 6:
                x += 1
            else:
                #print("Jedinec vysiel z mriezky D")
                jedinec.fitness -= 10
                x += 1
                break
            if mriezka[x][y] == 1:
                #print("Jedinec nasiel poklad")
                mriezka[x][y] = 0
                jedinec.fitness += 30
                jedinec.pocet_pokladov += 1
        elif krok == "P":
            jedinec.fitness -= 1
            if y < 6:
                y += 1
            else:
               # print("Jedinec vysiel z mriezky P")
                jedinec.fitness -= 10
                y += 1
                break
            if mriezka[x][y] == 1:
                #print("Jedinec nasiel poklad")
                mriezka[x][y] = 0
                jedinec.fitness += 30
                jedinec.pocet_pokladov += 1
        elif krok == "L":
            jedinec.fitness -= 1
            if y > 0:
                y -= 1
            else:
                #print("Jedinec vysiel z mriezky L")
                jedinec.fitness -= 10
                y -= 1
                break
            if mriezka[x][y] == 1:
                #print("Jedinec nasiel poklad")
                mriezka[x][y] = 0
                jedinec.fitness += 30
                jedinec.pocet_pokladov += 1

    if jedinec.pocet_pokladov == 5 and prvy_krat == 0:
        prvy_krat += 1
        return 1

    return 0

def krizenie(rodic1, rodic2):
    # Náhodne vyberieme bod rozdelenia medzi 1 a posledným indexom (nie začiatok ani koniec)
    bod_krizenia = random.randint(1, len(rodic1.hodnoty) - 1)

    # Vytvoríme dve rozdielne deti, ktoré berú hodnoty od svojich rodičov na základe náhodného bodu kríženia
    dieta1 = Jedinec()
    dieta1.hodnoty = rodic1.hodnoty[:bod_krizenia] + rodic2.hodnoty[bod_krizenia:]

    dieta2 = Jedinec()
    dieta2.hodnoty = rodic2.hodnoty[:bod_krizenia] + rodic1.hodnoty[bod_krizenia:]

    return dieta1, dieta2

def mutacia(original_jedinec, min_zmeny=1, max_zmeny=5):
    # Vytvor nového jedinca
    novy_jedinec = Jedinec()

    # Skopíruj hodnoty z pôvodného jedinca
    novy_jedinec.hodnoty = original_jedinec.hodnoty.copy()

    # Vyber náhodný počet zmien (od 1 do 5)
    pocet_zmien = random.randint(min_zmeny, max_zmeny)

    # Vyber náhodné pozície na zmenu
    pozicie_na_zmenu = random.sample(range(len(novy_jedinec.hodnoty)), pocet_zmien)

    # Na náhodných pozíciách nahraď hodnoty novými 8-bitovými binárnymi číslami
    for pozicia in pozicie_na_zmenu:
        novy_jedinec.hodnoty[pozicia] = format(random.randint(0, 255), '08b')

    return novy_jedinec

def vyber_top_jedincov(populacia, pocet):
    #Vyberie 'pocet' najlepších jedincov na základe fitness.
    populacia.sort(key=lambda jedinec: jedinec.fitness, reverse=True)  # Triedime populáciu podľa fitness od najlepšieho
    return populacia[:pocet]  # Vrátime prvých 'pocet' najlepších jedincov

def ruleta(populacia, pocet_jedincov):
    # Získanie fitness hodnôt pre každého jedinca
    fitness_hodnoty = [jedinec.fitness for jedinec in populacia]

    # Výber jedincov s pravdepodobnosťou podla fitness hodnoty
    vyber = random.choices(populacia, weights=fitness_hodnoty, k=pocet_jedincov)

    return vyber
#Grafika----------------------------------------------------------------------------------------------------------------------------------------------------------------
def vykonaj_instrukcie():
    #Vyzualizacia krokov
    global pohyb_instrukcie
    pohyb_instrukcie = list(instrukcie_sekvencia)
    pohyb_postavicky()

def pohyb_postavicky():
    #Animácia jedinca pohybujúceho sa na mriežke
    global postavicka_pozicia, pohyb_instrukcie

    if pohyb_instrukcie:
        instrukcia = pohyb_instrukcie.pop(0)  # Zober prvú inštrukciu

        # Pohyb postavičky
        if instrukcia == 'H':
            postavicka_pozicia[0] -= 1
        elif instrukcia == 'D':
            postavicka_pozicia[0] += 1
        elif instrukcia == 'P':
            postavicka_pozicia[1] += 1
        elif instrukcia == 'L':
            postavicka_pozicia[1] -= 1

        # Kontrola, či postavička opustila mriežku
        if postavicka_pozicia[0] < 0 or postavicka_pozicia[0] > 6 or postavicka_pozicia[1] < 0 or postavicka_pozicia[1] > 6:
            vystup_message1()
            return

        # Aktualizácia mriežky a znovu zavolanie pohybu
        vytvor_mriezku()
        root.after(500, pohyb_postavicky)  # rýchlosť pohybu

def vytvor_mriezku():
    #Vytvorenie mriežky
    global postavicka_pozicia

    # Vyčistenie mriežky
    for widget in root.winfo_children():
        widget.destroy()

    # Pozícia štartu
    start_pozicia = (6, 3)  # Rad 7, Stĺpec 4 (indexované od 0)

    # Pozície pokladov
    poklady = [(1, 4), (2, 2), (3, 6), (4, 1), (5, 4)]  # Všetko indexované od 0

    for row in range(7):
        for col in range(7):
            if (row, col) == tuple(postavicka_pozicia):
                farba = "blue"
                text = "Jedinec"
            elif (row, col) == start_pozicia:
                farba = "green"
                text = "Štart"
            elif (row, col) in poklady:
                farba = "yellow"
                text = "Poklad"
            else:
                farba = "white"
                text = ""

            # Vytvor štvorcový Label s nastavenou farbou a textom
            label = tk.Label(root, text=text, bg=farba, width=10, height=5, borderwidth=1, relief="solid")
            label.grid(row=row, column=col, padx=5, pady=5)

def vystup_message1():
    # Vypíš správu a ukonč program
    vystup_okno = tk.Toplevel(root)
    vystup_okno.title("Koniec hry")
    sprava = tk.Label(vystup_okno, text="Jedinec vyšiel z mriežky!")
    sprava.pack(pady=20)
#Grafika----------------------------------------------------------------------------------------------------------------------------------------------------------------
zacat_simulaciu = input("Zacat novu simulaciu?[ano/nie]:")
while zacat_simulaciu == "ano" or zacat_simulaciu == "Ano" or zacat_simulaciu == "ANO" or zacat_simulaciu == "a" or zacat_simulaciu == "A":
    generacia = 1 #aktualna generacia
    pocet_generacii = int(input("Kolko generacii?:"))
    prvy_krat = 0 #pouzite ked prvy jedinec najde vsetky poklady
    velkost_populacie = 100
    najdene_riesenie = 0 #kontrola ci sme nasli jedinca ktory pozbiera vsetky poklady
    start_time = time.time()# zaciatok casovania simulacie

    for i in range(pocet_generacii):
        #Ak niesme na prvej generacii
        if generacia > 1:
            # Reset krokov, počtu pokladov a počtu krokov pre všetkých jedincov
            for i in range(velkost_populacie):
                populacia[i].kroky = ['X' for i in range(500)]
                populacia[i].pocet_krokov = 0
                populacia[i].pocet_pokladov = 0

            # Zoradenie populácie podľa fitness od najlepšieho po najhoršieho
            populacia.sort(key=lambda jedinec: jedinec.fitness, reverse=True)

            # Vyberieme elitných jedincov
            elitni_jedinci = populacia[:1]

            # Vyberieme jedincov na mutáciu
            jedinci_na_mutaciu = ruleta(populacia, 48)
            jedinci_na_mutaciu.append(populacia[0])
            mutovani_jedinci = []
            for jedinec in jedinci_na_mutaciu:
                novy_mutovany = mutacia(jedinec)  # Vytvoríme mutovaného jedinca
                mutovani_jedinci.append(novy_mutovany)

            # Vyberieme rodičov pre kríženie
            rodicia = ruleta(populacia, 49)
            rodicia.append(populacia[0])
            # Vytvoríme novú populáciu
            nova_populacia = []

            # Pridáme elitných jedincov do novej populácie
            nova_populacia.extend(elitni_jedinci)

            # Pridáme mutovaných jedincov do novej populácie
            nova_populacia.extend(mutovani_jedinci)

            # Kríženie párov, každé kríženie pridá dve deti
            for j in range(0, 50, 2):
                rodic1 = rodicia[j]
                rodic2 = rodicia[j + 1]

                dieta1, dieta2 = krizenie(rodic1, rodic2)

                nova_populacia.append(dieta1)
                nova_populacia.append(dieta2)

            # Nahradíme starú populáciu novou populáciou
            populacia = nova_populacia

            for i in range(velkost_populacie):
                #reset na fitness
                populacia[i].fitness = 100
        else:
            # Vytvorí novú generáciu
            populacia = [Jedinec() for i in range(velkost_populacie)]

        for i in range(velkost_populacie):
            #generácia pohybov
            riesenie_jedinca(populacia[i], i)
        for i in range(velkost_populacie):
            #aplikuje pohyby (rieši)
            nasiel_vsetky = prechadzka(populacia[i], i, prvy_krat)
            if nasiel_vsetky == 1:
                najdene_riesenie += 1
        if najdene_riesenie == 1:
            print("Jedinec nasiel vsetky poklady")
            pokracovat = input("Pokracovat v simulacii?[ano/nie]:")
            if pokracovat == "nie" or pokracovat == "Nie" or pokracovat == "NIE" or pokracovat == "n" or pokracovat == "N":
                break
        if generacia % 100 == 0:
            print(f"{generacia} out of {pocet_generacii}")
            najlepsi_jedinec = vyber_top_jedincov(populacia, 1)[0]
            print(f"Fitness: {round(najlepsi_jedinec.fitness, 1)}")
            if najlepsi_jedinec.fitness == 234:
                break
        generacia += 1

    # Po poslednej generácii nájdeme a vypíšeme najlepšieho jedinca
    najlepsi_jedinec = vyber_top_jedincov(populacia, 1)[0]  # Vyberieme najlepšieho jedinca

    #Začiatočná pozícia postavičky pre animáciu (rovnaká ako pozícia štartu)
    postavicka_pozicia = [6, 3]
    pohyb_instrukcie = []
# Grafika----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Uloženie krokov do premennej vo formáte reťazca
    instrukcie_sekvencia = ''.join(najlepsi_jedinec.kroky)

    instrukcie_sekvencia = ''.join([krok for krok in najlepsi_jedinec.kroky if krok != 'X'])  # Vynechanie krokov 'X'
    pocet_krokov = len(instrukcie_sekvencia)  # Počet vykonaných krokov
    print("Top jedinec:")
    print(f"Fitness: {round(najlepsi_jedinec.fitness, 1)}")
    print(f"Kroky: {instrukcie_sekvencia}")
    print(f"Počet krokov: {pocet_krokov}")
    print(f"Počet nájdených pokladov: {najlepsi_jedinec.pocet_pokladov}")
    end_time = time.time() # koniec simulácie
    elapsed_time = end_time - start_time
    print(f"Simulácia bežala {round(elapsed_time,2)} sekúnd.")
    odpoved = input(f"Pozriet animaciu?[ano/nie]:")
    if odpoved == "ano" or odpoved == "Ano" or odpoved == "ANO" or odpoved == "a" or odpoved == "A":
        if __name__ == "__main__":
            root = tk.Tk()
            root.lift()
            root.attributes('-topmost', True)
            root.after(1000, lambda: root.attributes('-topmost', False))
            root.title("Mriežka 7x7 s postavičkou")
            vykonaj_instrukcie()  # Spusti inštrukcie hneď pri štarte
            root.mainloop()
# Grafika----------------------------------------------------------------------------------------------------------------------------------------------------------------
    zacat_simulaciu = input("Zacat novu simulaciu?[ano/nie]:")