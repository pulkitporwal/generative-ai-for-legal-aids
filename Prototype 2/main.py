import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

faiss_index_path = "faiss_index"

ollama_emb = OllamaEmbeddings(model="llama2:7b")

if os.path.exists(faiss_index_path):
    print("Loading existing FAISS index...")
    db = FAISS.load_local(faiss_index_path, ollama_emb, allow_dangerous_deserialization=True)

else:
    print("FAISS index not found, creating embeddings...")
    pdf_loader = PyPDFLoader(r"D:\Generative AI for Legal Aids\Prototype 2\data\bns.pdf")
    docs = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    db = FAISS.from_documents(documents=documents, embedding=ollama_emb)
    
    db.save_local(faiss_index_path)
    print("FAISS index created and saved locally.")

query = "A person hitted my friend by a rod then tell me which BNS section will be applied"
result = db.similarity_search(query)
# print(result[0].page_content)

llm = OllamaLLM(model="llama2:7b")
prompt = ChatPromptTemplate.from_template("""
                                          
BNS Section LIST:=
1. Short title, commencement and application.
2. Definitions.
3. General explanations.
CHAPTER II
OF PUNISHMENTS
4. Punishments.
5. Commutation of sentence.
6. Fractions of terms of punishment.
7. Sentence may be (in certain cases of imprisonment) wholly or partly rigorous or simple.
8. Amount of fine, liability in default of payment of fine, etc.
9. Limit of punishment of offence made up of several offences.
10. Punishment of person guilty of one of several offences, judgment stating that it is doubtful of 
which.
11. Solitary confinement.
12. Limit of solitary confinement.
13. Enhanced punishment for certain offences after previous conviction.
CHAPTER III
GENERAL EXCEPTIONS
14. Act done by a person bound, or by mistake of fact believing himself bound, by law.
15. Act of Judge when acting judicially.
16. Act done pursuant to judgment or order of Court.
17. Act done by a person justified, or by mistake of fact believing himself justified, by law.
18. Accident in doing a lawful act.
19. Act likely to cause harm, but done without criminal intent, and to prevent other harm.
20. Act of a child under seven years of age. 
21. Act of a child above seven and under twelve years of age of immature understanding.
22. Act of a person of unsound mind. 
23. Act of a person incapable of judgment by reason of intoxication caused against his will.
24. Offence requiring a particular intent or knowledge committed by one who is intoxicated.
2
SECTIONS
25. Act not intended and not known to be likely to cause death or grievous hurt, done by consent.
26. Act not intended to cause death, done by consent in good faith for person's benefit.
27. Act done in good faith for benefit of child or person of unsound mind, by, or by consent of 
guardian.
28. Consent known to be given under fear or misconception.
29. Exclusion of acts which are offences independently of harm caused.
30. Act done in good faith for benefit of a person without consent.
31. Communication made in good faith.
32. Act to which a person is compelled by threats.
33. Act causing slight harm.
Of right of private defence
34. Things done in private defence.
35. Right of private defence of body and of property.
36. Right of private defence against act of a person of unsound mind, etc.
37. Acts against which there is no right of private defence.
38. When right of private defence of body extends to causing death.
39. When such right extends to causing any harm other than death.
40. Commencement and continuance of right of private defence of body.
41. When right of private defence of property extends to causing death.
42. When such right extends to causing any harm other than death.
43. Commencement and continuance of right of private defence of property. 
44. Right of private defence against deadly assault when there is risk of harm to innocent person.
CHAPTER IV
OF ABETMENT, CRIMINAL CONSPIRACY AND ATTEMPT
of abetment
45. Abetment of a thing.
46. Abettor.
47. Abetment in India of offences outside India.
48. Abetment outside India for offence in India.
49. Punishment of abetment if act abetted is committed in consequence and where no express 
provision is made for its punishment.
50. Punishment of abetment if person abetted does act with different intention from that of 
abettor.
51. Liability of abettor when one act abetted and different act done.
52. Abettor when liable to cumulative punishment for act abetted and for act done.
53. Liability of abettor for an effect caused by act abetted different from that intended by abettor.
54. Abettor present when offence is committed.
3
SECTIONS
55. Abetment of offence punishable with death or imprisonment for life.
56. Abetment of offence punishable with imprisonment.
57. Abetting commission of offence by public or by more than ten persons.
58. Concealing design to commit offence punishable with death or imprisonment for life.
59. Public servant concealing design to commit offence which it is his duty to prevent.
60. Concealing design to commit offence punishable with imprisonment.
Of criminal conspiracy
61. Criminal conspiracy.
Of attempt
62. Punishment for attempting to commit offences punishable with imprisonment for life or other 
imprisonment.
CHAPTER V
OF OFFENCES AGAINST WOMAN AND CHILD
Of sexual offences
63. Rape.
64. Punishment for rape.
65. Punishment for rape in certain cases.
66. Punishment for causing death or resulting in persistent vegetative state of victim.
67. Sexual intercourse by husband upon his wife during separation.
68. Sexual intercourse by a person in authority. 
69. Sexual intercourse by employing deceitful means, etc.
70. Gang rape.
71. Punishment for repeat offenders.
72. Disclosure of identity of victim of certain offences, etc.
73. Printing or publishing any matter relating to Court proceedings without permission.
Of criminal force and assault against woman
74. Assault or use of criminal force to woman with intent to outrage her modesty.
75. Sexual harassment.
76. Assault or use of criminal force to woman with intent to disrobe.
77. Voyeurism.
78. Stalking.
79. Word, gesture or act intended to insult modesty of a woman.
Of offences relating to marriage
80. Dowry death.
4
SECTIONS
81. Cohabitation caused by man deceitfully inducing belief of lawful marriage.
82. Marrying again during lifetime of husband or wife.
83. Marriage ceremony fraudulently gone through without lawful marriage.
84. Enticing or taking away or detaining with criminal intent a married woman.
85. Husband or relative of husband of a woman subjecting her to cruelty.
86. Cruelty defined.
87. Kidnapping, abducting or inducing woman to compel her marriage, etc.
Of causing miscarriage, etc.
88. Causing miscarriage.
89. Causing miscarriage without woman's consent.
90. Death caused by act done with intent to cause miscarriage.
91. Act done with intent to prevent child being born alive or to cause to die after birth.
92. Causing death of quick unborn child by act amounting to culpable homicide.
Of offences against child
93. Exposure and abandonment of child under twelve years of age, by parent or person having 
care of it.
94. Concealment of birth by secret disposal of dead body. 
95. Hiring, employing or engaging a child to commit an offence.
96. Procuration of child.
97. Kidnapping or abducting child under ten years of age with intent to steal from its person.
98. Selling child for purposes of prostitution, etc.
99. Buying child for purposes of prostitution, etc.
CHAPTER VI
OF OFFENCES AFFECTING THE HUMAN BODY
Of offences affecting life
100. Culpable homicide.
101. Murder.
102. Culpable homicide by causing death of person other than person whose death was intended.
103. Punishment for murder.
104. Punishment for murder by life-convict.
105. Punishment for culpable homicide not amounting to murder.
106. Causing death by negligence.
107. Abetment of suicide of child or person of unsound mind.
108. Abetment of suicide.
109. Attempt to murder.
110. Attempt to commit culpable homicide.
5
SECTIONS
111. Organised crime.
112. Petty organised crime.
113. Terrorist act.
Of hurt
114. Hurt.
115. Voluntarily causing hurt.
116. Grievous hurt.
117. Voluntarily causing grievous hurt.
118. Voluntarily causing hurt or grievous hurt by dangerous weapons or means.
119. Voluntarily causing hurt or grievous hurt to extort property, or to constrain to an illegal act. 
120. Voluntarily causing hurt or grievous hurt to extort confession, or to compel restoration of 
property.
121. Voluntarily causing hurt or grievous hurt to deter public servant from his duty.
122. Voluntarily causing hurt or grievous hurt on provocation.
123. Causing hurt by means of poison, etc., with intent to commit an offence.
124. Voluntarily causing grievous hurt by use of acid, etc.
125. Act endangering life or personal safety of others.
Of wrongful restraint and wrongful confinement
126. Wrongful restraint.
127. Wrongful confinement.
Of criminal force and assault
128. Force.
129. Criminal force.
130. Assault.
131. Punishment for assault or criminal force otherwise than on grave provocation.
132. Assault or criminal force to deter public servant from discharge of his duty.
133. Assault or criminal force with intent to dishonour person, otherwise than on grave 
provocation.
134. Assault or criminal force in attempt to commit theft of property carried by a person.
135. Assault or criminal force in attempt to wrongfully confine a person. 
136. Assault or criminal force on grave provocation.
Of kidnapping, abduction, slavery and forced labour
137. Kidnapping.
138. Abduction.
139. Kidnapping or maiming a child for purposes of begging.
140. Kidnapping or abducting in order to murder or for ransom, etc.
141. Importation of girl or boy from foreign country.
6
SECTIONS
142. Wrongfully concealing or keeping in confinement, kidnapped or abducted person.
143. Trafficking of person.
144. Exploitation of a trafficked person.
145. Habitual dealing in slaves.
146. Unlawful compulsory labour.
CHAPTER VII
OF OFFENCES AGAINST THE STATE
147. Waging, or attempting to wage war, or abetting waging of war, against Government of 
India.
148. Conspiracy to commit offences punishable by section 147.
149. Collecting arms, etc., with intention of waging war against Government of India.
150. Concealing with intent to facilitate design to wage war.
151. Assaulting President, Governor, etc., with intent to compel or restrain exercise of any lawful 
power.
152. Act endangering sovereignty, unity and integrity of India. 
153. Waging war against Government of any foreign State at peace with Government of India.
154. Committing depredation on territories of foreign State at peace with Government of India.
155. Receiving property taken by war or depredation mentioned in sections 153 and 154.
156. Public servant voluntarily allowing prisoner of State or war to escape.
157. Public servant negligently suffering such prisoner to escape.
158. Aiding escape of, rescuing or harbouring such prisoner.
CHAPTER VIII
OF OFFENCES RELATING TO THE ARMY, NAVY AND AIR FORCE
159. Abetting mutiny, or attempting to seduce a soldier, sailor or airman from his duty.
160. Abetment of mutiny, if mutiny is committed in consequence thereof.
161. Abetment of assault by soldier, sailor or airman on his superior officer, when in execution of 
his office. 
162. Abetment of such assault, if assault committed.
163. Abetment of desertion of soldier, sailor or airman.
164. Harbouring deserter.
165. Deserter concealed on board merchant vessel through negligence of master.
166. Abetment of act of insubordination by soldier, sailor or airman.
167. Persons subject to certain Acts.
168. Wearing garb or carrying token used by soldier, sailor or airman.
CHAPTER IX
OF OFFENCES RELATING TO ELECTIONS
169. Candidate, electoral right defined.
7
SECTIONS
170. Bribery.
171. Undue influence at elections.
172. Personation at elections.
173. Punishment for bribery.
174. Punishment for undue influence or personation at an election.
175. False statement in connection with an election.
176. Illegal payments in connection with an election.
177. Failure to keep election accounts.
CHAPTER X
OF OFFENCES RELATING TO COIN, CURRENCY-NOTES, BANK-NOTES, AND GOVERNMENT STAMPS
178. Counterfeiting coin, Government stamps, currency-notes or bank-notes.
179. Using as genuine, forged or counterfeit coin, Government stamp, currency-notes or 
bank-notes.
180. Possession of forged or counterfeit coin, Government stamp, currency-notes or bank-notes.
181. Making or possessing instruments or materials for forging or counterfeiting coin, 
Government stamp, currency-notes or bank-notes.
182. Making or using documents resembling currency-notes or bank-notes. 
183. Effacing writing from substance bearing Government stamp, or removing from document a 
stamp used for it, with intent to cause loss to Government.
184. Using Government stamp known to have been before used.
185. Erasure of mark denoting that stamp has been used. 
186. Prohibition of fictitious stamps. 
187. Person employed in mint causing coin to be of different weight or composition from that 
fixed by law. 
188. Unlawfully taking coining instrument from mint.
CHAPTER XI
OF OFFENCES AGAINST THE PUBLIC TRANQUILLITY
189. Unlawful assembly.
190. Every member of unlawful assembly guilty of offence committed in prosecution of common 
object.
191. Rioting.
192. Wantonly giving provocation with intent to cause riot-if rioting be committed; if not 
committed.
193. Liability of owner, occupier, etc., of land on which an unlawful assembly or riot takes place. 
194. Affray. 
195. Assaulting or obstructing public servant when suppressing riot, etc.
196. Promoting enmity between different groups on grounds of religion, race, place of birth, 
residence, language, etc., and doing acts prejudicial to maintenance of harmony.
8
SECTIONS
197. Imputations, assertions prejudicial to national integration.
CHAPTER XII
OF OFFENCES BY OR RELATING TO PUBLIC SERVANTS
198. Public servant disobeying law, with intent to cause injury to any person.
199. Public servant disobeying direction under law.
200. Punishment for non-treatment of victim.
201. Public servant framing an incorrect document with intent to cause injury.
202. Public servant unlawfully engaging in trade.
203. Public servant unlawfully buying or bidding for property.
204. Personating a public servant.
205. Wearing garb or carrying token used by public servant with fraudulent intent.
CHAPTER XIII
OF CONTEMPTS OF THE LAWFUL AUTHORITY OF PUBLIC SERVANTS
206. Absconding to avoid service of summons or other proceeding.
207. Preventing service of summons or other proceeding, or preventing publication thereof.
208. Non-attendance in obedience to an order from public servant. 
209. Non-appearance in response to a proclamation under section 84 of Bharatiya Nagarik 
Suraksha Sanhita, 2023.
210. Omission to produce document or electronic record to public servant by person legally 
bound to produce it.
211. Omission to give notice or information to public servant by person legally bound to give it.
212. Furnishing false information.
213. Refusing oath or affirmation when duly required by public servant to make it.
214. Refusing to answer public servant authorised to question.
215. Refusing to sign statement.
216. False statement on oath or affirmation to public servant or person authorised to administer 
an oath or affirmation. 
217. False information, with intent to cause public servant to use his lawful power to injury of 
another person.
218. Resistance to taking of property by lawful authority of a public servant.
219. Obstructing sale of property offered for sale by authority of public servant. 
220. Illegal purchase or bid for property offered for sale by authority of public servant.
221. Obstructing public servant in discharge of public functions.
222. Omission to assist public servant when bound by law to give assistance.
223. Disobedience to order duly promulgated by public servant.
224. Threat of injury to public servant.
225. Threat of injury to induce person to refrain from applying for protection to public servant.
9
SECTIONS
226. Attempt to commit suicide to compel or restrain exercise of lawful power.
CHAPTER XIV
OF FALSE EVIDENCE AND OFFENCES AGAINST PUBLIC JUSTICE
227. Giving false evidence.
228. Fabricating false evidence.
229. Punishment for false evidence.
230. Giving or fabricating false evidence with intent to procure conviction of capital offence.
231. Giving or fabricating false evidence with intent to procure conviction of offence punishable 
with imprisonment for life or imprisonment.
232. Threatening any person to give false evidence.
233. Using evidence known to be false.
234. Issuing or signing false certificate.
235. Using as true a certificate known to be false.
236. False statement made in declaration which is by law receivable as evidence.
237. Using as true such declaration knowing it to be false.
238. Causing disappearance of evidence of offence, or giving false information to screen 
offender.
239. Intentional omission to give information of offence by person bound to inform.
240. Giving false information respecting an offence committed.
241. Destruction of document or electronic record to prevent its production as evidence.
242. False personation for purpose of act or proceeding in suit or prosecution. 
243. Fraudulent removal or concealment of property to prevent its seizure as forfeited or in 
execution.
244. Fraudulent claim to property to prevent its seizure as forfeited or in execution.
245. Fraudulently suffering decree for sum not due.
246. Dishonestly making false claim in Court.
247. Fraudulently obtaining decree for sum not due.
248. False charge of offence made with intent to injure.
249. Harbouring offender.
250. Taking gift, etc., to screen an offender from punishment.
251. Offering gift or restoration of property in consideration of screening offender.
252. Taking gift to help to recover stolen property, etc.
253. Harbouring offender who has escaped from custody or whose apprehension has been 
ordered.
254. Penalty for harbouring robbers or dacoits.
255. Public servant disobeying direction of law with intent to save person from punishment or 
property from forfeiture.
10
SECTIONS
256. Public servant framing incorrect record or writing with intent to save person from 
punishment or property from forfeiture.
257. Public servant in judicial proceeding corruptly making report, etc., contrary to law.
258. Commitment for trial or confinement by person having authority who knows that he is 
acting contrary to law.
259. Intentional omission to apprehend on part of public servant bound to apprehend.
260. Intentional omission to apprehend on part of public servant bound to apprehend person 
under sentence or lawfully committed.
261. Escape from confinement or custody negligently suffered by public servant.
262. Resistance or obstruction by a person to his lawful apprehension.
263. Resistance or obstruction to lawful apprehension of another person.
264. Omission to apprehend, or sufferance of escape, on part of public servant, in cases not 
otherwise provided for.
265. Resistance or obstruction to lawful apprehension or escape or rescue in cases not otherwise 
provided for.
266. Violation of condition of remission of punishment.
267. Intentional insult or interruption to public servant sitting in judicial proceeding.
268. Personation of assessor.
269. Failure by person released on bail bond or bond to appear in Court.
CHAPTER XV
OF OFFENCES AFFECTING THE PUBLIC HEALTH, SAFETY, CONVENIENCE, 
DECENCY AND MORALS
270. Public nuisance.
271. Negligent act likely to spread infection of disease dangerous to life.
272. Malignant act likely to spread infection of disease dangerous to life.
273. Disobedience to quarantine rule. 
274. Adulteration of food or drink intended for sale.
275. Sale of noxious food or drink.
276. Adulteration of drugs.
277. Sale of adulterated drugs.
278. Sale of drug as a different drug or preparation.
279. Fouling water of public spring or reservoir.
280. Making atmosphere noxious to health.
281. Rash driving or riding on a public way.
282. Rash navigation of vessel.
283. Exhibition of false light, mark or buoy.
284. Conveying person by water for hire in unsafe or overloaded vessel.
285. Danger or obstruction in public way or line of navigation.
11
SECTIONS
286. Negligent conduct with respect to poisonous substance.
287. Negligent conduct with respect to fire or combustible matter.
288. Negligent conduct with respect to explosive substance.
289. Negligent conduct with respect to machinery.
290. Negligent conduct with respect to pulling down, repairing or constructing buildings, etc.
291. Negligent conduct with respect to animal.
292. Punishment for public nuisance in cases not otherwise provided for.
293. Continuance of nuisance after injunction to discontinue.
294. Sale, etc., of obscene books, etc.
295. Sale, etc., of obscene objects to child.
296. Obscene acts and songs.
297. Keeping lottery office.
CHAPTER XVI
OF OFFENCES RELATING TO RELIGION
298. Injuring or defiling place of worship with intent to insult religion of any class.
299. Deliberate and malicious acts, intended to outrage religious feelings of any class by
insulting its religion or religious beliefs.
300. Disturbing religious assembly.
301. Trespassing on burial places, etc.
302. Uttering words, etc., with deliberate intent to wound religious feelings of any person.
CHAPTER XVII
OF OFFENCES AGAINST PROPERTY
Of theft
303. Theft.
304. Snatching. 
305. Theft in a dwelling house, or means of transportation or place of worship, etc.
306. Theft by clerk or servant of property in possession of master.
307. Theft after preparation made for causing death, hurt or restraint in order to committing of 
theft.
Of extortion
308. Extortion.
Of robbery and dacoity
309. Robbery.
310. Dacoity.
311. Robbery, or dacoity, with attempt to cause death or grievous hurt.
312. Attempt to commit robbery or dacoity when armed with deadly weapon.
313. Punishment for belonging to gang of robbers, etc.
12
Of criminal misappropriation of property
SECTIONS
314. Dishonest misappropriation of property.
315. Dishonest misappropriation of property possessed by deceased person at the time of his 
death.
Of criminal breach of trust
316. Criminal breach of trust.
Of receiving stolen property
317. Stolen property.
Of cheating
318. Cheating.
319. Cheating by personation.
Of fraudulent deeds and dispositions of property
320. Dishonest or fraudulent removal or concealment of property to prevent distribution among 
creditors.
321. Dishonestly or fraudulently preventing debt being available for creditors.
322. Dishonest or fraudulent execution of deed of transfer containing false statement of 
consideration.
323. Dishonest or fraudulent removal or concealment of property.
Of mischief
324. Mischief.
325. Mischief by killing or maiming animal.
326. Mischief by injury, inundation, fire or explosive substance, etc.
327. Mischief with intent to destroy or make unsafe a rail, aircraft, decked vessel or one of 
twenty tons burden.
328. Punishment for intentionally running vessel aground or ashore with intent to commit theft, 
etc.
Of criminal trespass
329. Criminal trespass and house-trespass.
330. House-trespass and hous-ebreaking.
331. Punishment for house-trespass or house-breaking.
332. House-trespass in order to commit offence.
333. House-trespass after preparation for hurt, assault or wrongful restraint.
334. Dishonestly breaking open receptacle containing property.
CHAPTER XVIII
OF OFFENCES RELATING TO DOCUMENTS AND TO PROPERTY MARKS
335. Making a false document.
13
SECTIONS
336. Forgery.
337. Forgery of record of Court or of public register, etc.
338. Forgery of valuable security, will, etc.
339. Having possession of document described in section 337 or section 338, knowing it to be 
forged and intending to use it as genuine.
340. Forged document or electronic record and using it as genuine.
341. Making or possessing counterfeit seal, etc., with intent to commit forgery punishable under 
section 338.
342. Counterfeiting device or mark used for authenticating documents described in section 338, 
or possessing counterfeit marked material.
343. Fraudulent cancellation, destruction, etc., of will, authority to adopt, or valuable security.
344. Falsification of accounts.
Of property marks
345. Property mark.
346. Tampering with property mark with intent to cause injury.
347. Counterfeiting a property mark.
348. Making or possession of any instrument for counterfeiting a property mark.
349. Selling goods marked with a counterfeit property mark.
350. Making a false mark upon any receptacle containing goods.
CHAPTER XIX
OF CRIMINAL INTIMIDATION, INSULT, ANNOYANCE, DEFAMATION, ETC.
351. Criminal intimidation.
352. Intentional insult with intent to provoke breach of peace.
353. Statements conducing to public mischief.
354. Act caused by inducing person to believe that he will be rendered an object of Divine 
displeasure.
355. Misconduct in public by a drunken person.
Of defamation
356. Defamation.
Of breach of contract to attend on and supply wants of helpless person
357. Breach of contract to attend on and supply wants of helpless person.
CHAPTER XX
REPEAL AND SAVINGS
358. Repeal and savings

You are an Expert Legal Advisor and you have to answer Bhartiya Nyay Sanhita Sections.
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "A man hit my friend with a rod. Tell me BNS Section"})
# print(response['answer'])

def rephrase_query(query):
    print(query)
    
    rephrase_prompt = PromptTemplate.from_template("""
    You are an expert legal translator, well-versed in the language and structure of Indian legal codes, particularly the Bhartiya Nyaya Sanhita (BNS). Your task is to transform informal descriptions of incidents or legal issues into formal, precise legal language that aligns closely with the terminology found in official legal documents.

    User Query = {query}

    Your task:
    1. Analyze the user's input to identify the core legal issues or relevant actions described.
    2. Translate the key elements of the situation into formal legal terminology, using language that would be found in the Bhartiya Nyaya Sanhita (BNS).
    3. Structure the output as a concise, single-sentence statement that encapsulates the legal essence of the situation.
    4. Ensure that the resulting statement uses terms and phrasings that are likely to appear in official legal documents, facilitating easier matching with relevant sections of the law.
    5. Avoid including specific names, dates, or locations in the output. Focus on the actions, intentions, and consequences described.
    6. Don't give me any sections. Just give me the sentence in and bridges the gap between colloquial descriptions and formal legal language.
    7. Don't add the IPC Sections

    Example input: "My neighbor's dog bit me when I was walking on the sidewalk in front of their house."

    Example output: "The incident involves the infliction of physical injury by a domesticated animal upon a person lawfully present in a public space adjacent to the animal owner's property."

    Remember: Your goal is to produce a statement that bridges the gap between colloquial descriptions and formal legal language, facilitating more accurate identification of relevant legal statutes and precedents.

    Return only the rephrased sentence without any additional information or formatting.
    """)
    
    rephrase_chain = LLMChain(llm=llm, prompt=rephrase_prompt)
    
    rephrased_response = rephrase_chain.invoke({"query": query})

    return rephrased_response['text'].strip() 

def generate_answer_with_rephrase(query):
    rephrased_query = rephrase_query(query)
    print(f"Rephrased Query: {rephrased_query}")

    result = db.similarity_search(rephrased_query)
    if not result:
        return "No relevant information found in the database."

    context = result[0].page_content

    response = retrieval_chain.invoke({"input": query, "context": context})

    return response.get('answer', 'No answer was generated.')

query = "A man hit my friend with a rod. Tell me BNS Section"
answer = generate_answer_with_rephrase(query)