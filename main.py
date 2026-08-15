from fastapi import FastAPI,HTTPException
from pydantic import BaseModel

from agents import ( build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain
    )

app = FastAPI(title="Multi Agent Research API",
              description="AI powered multi agent research system",
              version="1.0.0")


class ResearchRequest(BaseModel):
    topic:str

class ResearchResponse(BaseModel):
    topic :str
    research:str
    report:str
    critique:str

search_agent = build_search_agent()
reader_agent = build_reader_agent()

@app.get("/")
def root():
    return{
        "message":"Multi Agent Research API is running"
    }

@app.get("/health")
def health():
    return{
        "status":"healthy"
    }

@app.post("/research",response_model=ResearchResponse)
def research(request:ResearchRequest):
    topic = request.topic.strip()

    if not topic:
        raise HTTPException(
            status_code=400,
            detail= "Research topic cannot be empty"
        )
    try:
        search_result = search_agent.invoke({
            "messages":[
                {
                    "role":"user",
                    "content":(
                        f"Research the following topic thoroughly:{topic}"
                        "Find recent,reliable and relevent sources"
                        "Return the important findings and URLs"
                    )
                }
            ]
        })
        research = search_result["messages"][-1].content

        reader_result= reader_agent.invoke({
            "messages":[
                {
                    "role":"user",
                    "content":(
                        f"Read and analyze the following research results"
                        f"for the topic: {topic}\n\n"
                        f"{research}\n\n"
                        "Use the URLs from the research to obtain deeper "
                        "information where useful. Extract important facts, "
                        "evidence and source information."
                    )
                }
            ]
        })
        research += "\n\nDETAILED SOURCE RESEARCH:\n"
        research += reader_result["messages"][-1].content

        report = writer_chain.invoke({
            "topic":topic,
            "research":research
        })

        critique = critic_chain.invoke({
            "report":report
        })

        return ResearchResponse(
            topic=topic,
            research=research,
            report=report,
            critique=critique
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Research pipeline failed:{str(e)}"
        )